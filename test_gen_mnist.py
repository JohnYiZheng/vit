import os
import sys
import torch
import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw
root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_path)

from vit_torch import MyViT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyViT(
        (1, 28, 28), n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, out_d=10
    ).to(device)
model.load_state_dict(torch.load("vit_ep4.pth"))

image = None
# 创建推理窗口
def create_inference_window():
    global image
    inference_window = tk.Tk()
    inference_window.title("手写数字识别")

    # 创建画布
    canvas = tk.Canvas(inference_window, width=280, height=280, bg="black")
    canvas.pack()

    # 创建画笔
    image = Image.new("L", (280, 280), "black")
    draw = ImageDraw.Draw(image)

    # 记录鼠标位置
    last_x, last_y = None, None

    # 鼠标按下事件
    def on_mouse_down(event):
        global last_x, last_y
        last_x, last_y = event.x, event.y

    # 鼠标移动事件
    def on_mouse_move(event):
        global last_x, last_y
        if last_x and last_y:
            canvas.create_line((last_x, last_y, event.x, event.y), width=15, fill="white", capstyle=tk.ROUND, smooth=tk.TRUE)
            draw.line((last_x, last_y, event.x, event.y), fill="white", width=15)
            last_x, last_y = event.x, event.y

    # 鼠标释放事件
    def on_mouse_up(event):
        global last_x, last_y
        last_x, last_y = None, None

    # 清除画布
    def clear_canvas():
        canvas.delete("all")
        draw.rectangle((0, 0, 280, 280), fill="black")

    # 进行推理
    def inference():
        global image
        # 调整图像大小为28x28
        resized_image = image.resize((28, 28))
        data = np.array(resized_image).astype("float32") / 255.0
        print(data)
        data_tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            prediction = model.forward(data_tensor)
            print("prediction logits:", prediction)
        result_label.config(text=f"预测结果：{np.argmax(prediction.cpu().detach().numpy())}")

    # 绑定事件
    canvas.bind("<Button-1>", on_mouse_down)
    canvas.bind("<B1-Motion>", on_mouse_move)
    canvas.bind("<ButtonRelease-1>", on_mouse_up)

    # 添加清除按钮和推理按钮
    clear_button = tk.Button(inference_window, text="清除", command=clear_canvas)
    clear_button.pack(side=tk.LEFT, padx=10)
    infer_button = tk.Button(inference_window, text="识别", command=inference)
    infer_button.pack(side=tk.LEFT, padx=10)
    result_label = tk.Label(inference_window, text="")
    result_label.pack()

    # 运行主循环
    inference_window.mainloop()

create_inference_window()