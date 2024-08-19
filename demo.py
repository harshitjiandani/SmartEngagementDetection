import argparse
import os
import uuid
import io
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.responses import StreamingResponse
from PIL import Image
import uvicorn
from predict import pred
import numpy as np
import cv2

ypred="n.a"
# Define the FastAPI application
app = FastAPI()
# Define your upload handling route
@app.get("/")
async def form():
    content = """
    <body>
    <h3>Upload Image for engagement detection</h3>
    <form action="/upload/" enctype="multipart/form-data" method="post">
        <input name="image" type="file" accept="image/*">
        <br>
        <button type="submit">Upload</button>
    </form>
</body>
    """
    return HTMLResponse(content=content)
@app.post("/upload/")
async def handle_uploads(image: UploadFile = File(...)):
    try:

        if not image.content_type.startswith('image/'):
            return {"message": "Invalid image file type. Please upload an image."}
        
        contentss = await image.read()
        img = Image.open(io.BytesIO(contentss))

        # Convert PIL image to NumPy array
        img_np = np.array(img)
        
        
        
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        print(img_np.shape)
        y_pred = pred(img_np)

        success_message = "Files processed successfully"
        video_button = '<form action="/view-engagement" method="get"><button type="submit">View Result</button></form>'
        return HTMLResponse(content=f"{success_message}<br>{ypred}")
    
    except Exception as e:
        print(f"Error uploading or processing files: {e}")
        return {"message": "There was an error during the process."}


def main(host: str, port: int):
    # Run the FastAPI server with specified host and port
    uvicorn.run(app, host=host, port=port)


# Define argument parsing
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Start the FastAPI server.")
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to run the FastAPI server')
    parser.add_argument('--port', type=int, default=8000, help='Port to run the FastAPI server')

    args = parser.parse_args()
    main(host=args.host, port=args.port)
