import base64
from email.mime import message
import cv2
from fastapi.concurrency import asynccontextmanager
import numpy as np
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.utils import build_inference_frame, make_robot_action
from openai import OpenAI
import asyncio
from fastapi import BackgroundTasks



# For LMstudio api
model_name = ""
client = OpenAI(
    base_url="",
    api_key="",
)

# Variables for task tracking
task_in_progress = False
real_task = ""
latest_top_image = ""

# Async function to monitor task completion
async def monitor_task_completion():
    global task_in_progress, real_task, latest_top_image

    await asyncio.sleep(5) 

    # While theres a task in progrss if theres an image, a call is sent to the api
    while task_in_progress:
        if latest_top_image:
            try:
                # Run the synchronous OpenAI call in a thread so it doesn't freeze FastAPI
                completion = await asyncio.to_thread(
                    client.chat.completions.create,
                    model=model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": f"Has the task '{real_task}' been completed? respond with Yes or No only"},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{latest_top_image}"}}
                            ]
                        }
                    ],
                    temperature=0.0
                )
                
                completion_text = completion.choices[0].message.content.strip().lower()
                print(f"[Judge]: Task '{real_task}' done? {completion_text}")

                # If the task is done we stop the current task
                if "yes" in completion_text:
                    task_in_progress = False
                    real_task = ""
                    break

            except Exception as e:
                print(f"[API Error]: {e}")
        
        # 3 second delay
        await asyncio.sleep(3)

# Old implementation
# def judge_task_done(judge_img):
#     global task_in_progress, real_task
#     try:
#         completion = client.chat.completions.create(
#             model=model_name,
#             messages=[
#                 {
#                     "role": "user",
#                     "content": [
#                         {"type": "text", "text": f"Has the task '{real_task}' been completed? respond with Yes or No only"},
#                         {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{judge_img}"}}
#                     ]
#                 }
#             ],
#         )
    
#         completion_text = completion.choices[0].message.content
#         if(completion_text.strip().lower() == "yes"):
#             task_in_progress = False
#             return True
#         else:
#             task_in_progress = True
#             return False
#     except Exception as e:
#         print(f"API Error: {e}")
#         # On API error assume task not done so the policy can continue
#         return False

# Smolvla variables
device = torch.device("cuda")
model_id = "lerobot/smolvla_base"
policy = None
preprocess = None
postprocess = None
    
@asynccontextmanager
async def lifespan(app: FastAPI):
    global policy, preprocess, postprocess

    policy = SmolVLAPolicy.from_pretrained(model_id).to(device).eval()

    preprocess, postprocess = make_smolvla_pre_post_processors(
        policy.config,
        # policy.config.__dict__,
        None,
    )
    yield


app = FastAPI(title="LeRobot Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Structure of request payloads from the Robot
class InferenceRequest(BaseModel):
    image_top64: str
    image_head64: str
    image_side64: str
    # task: str = "pick up the object"
    joint_states: list[float]

# Defining predict endpoint
@app.options("/predict")
def predict_options():
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        },
    )

# Defining set task endpoint
@app.post("/set_task")
def set_task(task: str, background_tasks: BackgroundTasks):
    global real_task, task_in_progress

    # Stops overwrite of tasks before they're completed
    if task_in_progress == True:
        return {"status": "error", "message": "task in progress"}
    else:
        real_task = task
        task_in_progress = True

        background_tasks.add_task(monitor_task_completion)
        return {"status": "success", "message": f"Task set to: {task}"}
    

@app.post("/predict")
def predict_action(req: InferenceRequest):
    global latest_top_image

    latest_top_image = req.image_top64

    if not task_in_progress:
        return {
            "status": "Task Done",
            "action": [],
            "message": "Current Task finished",
        }
    
    try:
       
        # Decode Base64 string to bytes
        img_top_bytes = base64.b64decode(req.image_top64)
        img_head_bytes = base64.b64decode(req.image_head64)
        img_side_bytes = base64.b64decode(req.image_side64)

        # Convert bytes to numpy arrays
        nparr_top = np.frombuffer(img_top_bytes, np.uint8)
        nparr_head = np.frombuffer(img_head_bytes, np.uint8)
        nparr_side = np.frombuffer(img_side_bytes, np.uint8)

        # Decode image array to OpenCV matrix
        cv_img_top = cv2.imdecode(nparr_top, cv2.IMREAD_COLOR)
        cv_img_head = cv2.imdecode(nparr_head, cv2.IMREAD_COLOR)
        cv_img_side = cv2.imdecode(nparr_side, cv2.IMREAD_COLOR)

        # if cv_img_top is None or cv_img_head is None or cv_img_side is None:
        #     raise ValueError("Failed to decode one or both images")

        rgb_top = cv2.cvtColor(cv_img_top, cv2.COLOR_BGR2RGB)
        rgb_head = cv2.cvtColor(cv_img_head, cv2.COLOR_BGR2RGB)
        rgb_side = cv2.cvtColor(cv_img_side, cv2.COLOR_BGR2RGB)

        tensor_top = (
            torch.from_numpy(rgb_top)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device=device, dtype=torch.float32) / 255.0
        )

        tensor_head = (
            torch.from_numpy(rgb_head)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device=device, dtype=torch.float32) / 255.0
        )

        tensor_side = (
            torch.from_numpy(rgb_side)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device=device, dtype=torch.float32) / 255.0
        )

        state_tensor = (
        torch.tensor(req.joint_states, dtype=torch.float32)
        .unsqueeze(0)
        .to(device)
        )

        # Payload for the policy
        transition = {
            "observation.images.camera1": tensor_top,
            "observation.images.camera2": tensor_head,
            "observation.images.camera3": tensor_side,
            "observation.state": state_tensor,
            "task": real_task,
        }

        transition = preprocess(transition)

        action = policy.select_action(transition)
        action = postprocess(action)
        action_list = action.squeeze(0).cpu().tolist()

        # Old code for displaying camera feeds in sim
        # cv2.imshow("ROS Top Camera Test", cv_img_top)
        # cv2.imshow("ROS Side Camera Test", cv_img_side)
        # cv2.waitKey(5)

        
        return {
            "status": "success",
            "action": action_list,
        }

    except Exception as e:
        print(f"[ERROR] {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)