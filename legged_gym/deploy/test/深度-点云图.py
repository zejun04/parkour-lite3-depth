import mujoco
import mujoco.viewer as viewer
import numpy as np
import glfw
import cv2
##
def glFrustum_CD_float32(znear, zfar):
  zfar  = np.float32(zfar)
  znear = np.float32(znear)
  C = -(zfar + znear)/(zfar - znear)
  D = -(np.float32(2)*zfar*znear)/(zfar - znear)
  return C, D
##
def ogl_zbuf_projection_inverse(zbuf, C, D):
  zlinear = 1 / ((zbuf - (-C)) / D) # TODO why -C?
  return zlinear

##
def ogl_zbuf_default_inv(zbuf_scaled, znear=None, zfar=None, C=None, D=None):
  if C is None:
    C, D = glFrustum_CD_float32(znear, zfar)
  zbuf = 2.0 * zbuf_scaled - 1.0
  zlinear = ogl_zbuf_projection_inverse(zbuf, C, D)
  return zlinear

##
def ogl_zbuf_negz_inv(zbuf, znear=None, zfar=None, C=None, D=None):
  if C is None:
    C, D = glFrustum_CD_float32(znear, zfar)
    C = np.float32(-0.5)*C - np.float32(0.5)
    D = np.float32(-0.5)*D
  zlinear = ogl_zbuf_projection_inverse(zbuf, C, D)
  return zlinear

# 设置分辨率
resolution = (640, 480)

# 创建OpenGL上下文（离屏渲染）
glfw.init()
glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
window = glfw.create_window(resolution[0], resolution[1], "Offscreen", None, None)
glfw.make_context_current(window)

# 加载MuJoCo模型
model = mujoco.MjModel.from_xml_path('urdf.xml')
data = mujoco.MjData(model)
scene = mujoco.MjvScene(model, maxgeom=10000)
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

# set camera properties
camera_name = "depth_camera"
camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
camera = mujoco.MjvCamera()
camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
if camera_id != -1:
    print("camera_id", camera_id)
    camera.fixedcamid = camera_id


# 创建帧缓冲对象并启用离屏渲染
framebuffer = mujoco.MjrRect(0, 0, resolution[0], resolution[1])
mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, context)

while True:
    mujoco.mj_step(model, data)
    
    # 获取目标物体的ID并设置相机跟踪该物体
    tracking_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    camera.trackbodyid = tracking_body_id
    camera.distance = 1.5  # 相机与目标的距离
    camera.azimuth = 0     # 水平方位角
    camera.elevation = -45  # 俯仰角 NOTE 顺时针：-90-----0-----90

    # 更新场景
    viewport = mujoco.MjrRect(0, 0, resolution[0], resolution[1])
    mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), 
                         mujoco.MjvPerturb(), camera, 
                         mujoco.mjtCatBit.mjCAT_ALL, scene)
    
    # 渲染场景并读取像素数据（RGB）
    mujoco.mjr_render(viewport, scene, context)
    rgb = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
    depth_buffer = np.zeros((resolution[1], resolution[0], 1), dtype=np.float32) ## 深度图像
    mujoco.mjr_readPixels(rgb, depth_buffer, viewport, context)


    # 转换颜色空间 (OpenCV使用BGR格式)
    bgr = cv2.cvtColor(np.flipud(rgb), cv2.COLOR_RGB2BGR)
    cv2.imshow('MuJoCo Camera Output', bgr)
    depth_buffer = np.flip(depth_buffer, axis=0).squeeze() ## 需要和OPENGL里面的颠倒方向，不然方向是反的

    
    # depth_image = (depth_buffer - np.min(depth_buffer)) / (np.max(depth_buffer) - np.min(depth_buffer))  # 归一化深度数据
    # depth_image = np.uint8(depth_image * 255)  # 转换为8位图像
    # depth_bgr = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)  # 使用JET色图显示深度
    # cv2.imshow('Depth Map', depth_bgr)

    ## 归一化
    depth_normalized = (depth_buffer - np.min(depth_buffer)) / (np.max(depth_buffer) - np.min(depth_buffer))
    depth_grayscale = np.uint8(depth_normalized * 255)
    cv2.imshow('Depth Map (Grayscale)', depth_grayscale)
    # depth_image = depth_image.astype(np.float64)
    # zfar  = model.vis.map.zfar * model.stat.extent
    # znear = model.vis.map.znear * model.stat.extent
    # depth_hat = ogl_zbuf_negz_inv(depth_image, znear, zfar)
    # print(depth_hat)

    
    
    # 退出条件（按Esc键退出）
    if cv2.waitKey(1) == 27:
        break

# 保存最后一帧的图像
cv2.imwrite('debug_output.png', bgr)

# 退出OpenCV和MuJoCo
cv2.destroyAllWindows()
glfw.terminate()
del context
