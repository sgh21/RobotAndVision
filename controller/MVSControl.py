import cv2
import numpy as np
from ctypes import *
from MvImport.MvCameraControl_class import *
from MvImport.PixelType_header import *

class MVSController:
    def __init__(self):
        self.cam = None
        self.data_buf = None
        self.nPayloadSize = None
        self.init()

    def enum_device(self, tlayerType, deviceList):
        """
        ch:枚举设备 | en:Enum device
        nTLayerType [IN] 枚举传输层 ，pstDevList [OUT] 设备列表
        """
        ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
        if ret != 0:
            print("enum devices fail! ret[0x%x]" % ret)
            sys.exit()

        if deviceList.nDeviceNum == 0:
            print("find no device!")
            sys.exit()

        print("Find %d devices!" % deviceList.nDeviceNum)

        for i in range(0, deviceList.nDeviceNum):
            mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
                print("\ngige device: [%d]" % i)
                # 输出设备名字
                strModeName = ""
                for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName:
                    strModeName = strModeName + chr(per)
                print("device model name: %s" % strModeName)
                # 输出设备ID
                nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
                nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
                nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
                nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
                print("current ip: %d.%d.%d.%d\n" % (nip1, nip2, nip3, nip4))
            # 输出USB接口的信息
            elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
                print("\nu3v device: [%d]" % i)
                strModeName = ""
                for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName:
                    if per == 0:
                        break
                    strModeName = strModeName + chr(per)
                print("device model name: %s" % strModeName)

                strSerialNumber = ""
                for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                    if per == 0:
                        break
                    strSerialNumber = strSerialNumber + chr(per)
                print("user serial number: %s" % strSerialNumber)

    # 判读图像格式是彩色还是黑白
    def is_image_color(self, enType):
        dates = {
            PixelType_Gvsp_RGB8_Packed: 'color',
            PixelType_Gvsp_BGR8_Packed: 'color',
            PixelType_Gvsp_YUV422_Packed: 'color',
            PixelType_Gvsp_YUV422_YUYV_Packed: 'color',
            PixelType_Gvsp_BayerGR8: 'color',
            PixelType_Gvsp_BayerRG8: 'color',
            PixelType_Gvsp_BayerGB8: 'color',
            PixelType_Gvsp_BayerBG8: 'color',
            PixelType_Gvsp_BayerGB10: 'color',
            PixelType_Gvsp_BayerGB10_Packed: 'color',
            PixelType_Gvsp_BayerBG10: 'color',
            PixelType_Gvsp_BayerBG10_Packed: 'color',
            PixelType_Gvsp_BayerRG10: 'color',
            PixelType_Gvsp_BayerRG10_Packed: 'color',
            PixelType_Gvsp_BayerGR10: 'color',
            PixelType_Gvsp_BayerGR10_Packed: 'color',
            PixelType_Gvsp_BayerGB12: 'color',
            PixelType_Gvsp_BayerGB12_Packed: 'color',
            PixelType_Gvsp_BayerBG12: 'color',
            PixelType_Gvsp_BayerBG12_Packed: 'color',
            PixelType_Gvsp_BayerRG12: 'color',
            PixelType_Gvsp_BayerRG12_Packed: 'color',
            PixelType_Gvsp_BayerGR12: 'color',
            PixelType_Gvsp_BayerGR12_Packed: 'color',
            PixelType_Gvsp_Mono8: 'mono',
            PixelType_Gvsp_Mono10: 'mono',
            PixelType_Gvsp_Mono10_Packed: 'mono',
            PixelType_Gvsp_Mono12: 'mono',
            PixelType_Gvsp_Mono12_Packed: 'mono'}
        return dates.get(enType, '未知')

    def enable_device(self, nConnectionNum, deviceList):
        """
        设备使能
        :param nConnectionNum: 设备编号
        :return: 相机, 图像缓存区, 图像数据大小
        """
        # ch:创建相机实例 | en:Creat Camera Object
        self.cam = MvCamera()

        # ch:选择设备并创建句柄 | en:Select device and create handle
        # cast(typ, val)，这个函数是为了检查val变量是typ类型的，但是这个cast函数不做检查，直接返回val
        stDeviceList = cast(deviceList.pDeviceInfo[int(nConnectionNum)], POINTER(MV_CC_DEVICE_INFO)).contents

        ret = self.cam.MV_CC_CreateHandle(stDeviceList)
        if ret != 0:
            print("create handle fail! ret[0x%x]" % ret)
            sys.exit()

        # ch:打开设备 | en:Open device
        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            print("open device fail! ret[0x%x]" % ret)
            sys.exit()

        # ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
        if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
            nPacketSize = self.cam.MV_CC_GetOptimalPacketSize()
            if int(nPacketSize) > 0:
                ret = self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
                if ret != 0:
                    print("Warning: Set Packet Size fail! ret[0x%x]" % ret)
            else:
                print("Warning: Get Packet Size fail! ret[0x%x]" % nPacketSize)

        # ch:设置触发模式为off | en:Set trigger mode as off
        ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        if ret != 0:
            print("set trigger mode fail! ret[0x%x]" % ret)
            sys.exit()

        # 从这开始，获取图片数据
        # ch:获取数据包大小 | en:Get payload size
        stParam = MVCC_INTVALUE()
        memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))
        # MV_CC_GetIntValue，获取Integer属性值，handle [IN] 设备句柄
        # strKey [IN] 属性键值，如获取宽度信息则为"Width"
        # pIntValue [IN][OUT] 返回给调用者有关相机属性结构体指针
        # 得到图片尺寸，这一句很关键
        # payloadsize，为流通道上的每个图像传输的最大字节数，相机的PayloadSize的典型值是(宽x高x像素大小)，此时图像没有附加任何额外信息
        ret = self.cam.MV_CC_GetIntValue("PayloadSize", stParam)
        if ret != 0:
            print("get payload size fail! ret[0x%x]" % ret)
            sys.exit()

        self.nPayloadSize = stParam.nCurValue

        # ch:开始取流 | en:Start grab image
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            print("start grabbing fail! ret[0x%x]" % ret)
            sys.exit()
        #  返回获取图像缓存区。
        self.data_buf = (c_ubyte * self.nPayloadSize)()
        #  date_buf前面的转化不用，不然报错，因为转了是浮点型
        return self.cam, self.data_buf, self.nPayloadSize
    
    def convert_to_opencv_format(self, data, height, width, pixel_format):
        """
        将相机不同的像素格式转换为OpenCV标准格式（BGR或灰度）
        
        参数:
        data -- 原始图像数据
        height -- 图像高度
        width -- 图像宽度
        pixel_format -- 像素格式，来自PixelType_header中的常量
        
        返回:
        OpenCV格式的图像（BGR或灰度）
        """
        # 判断是否为彩色图像
        is_color = self.is_image_color(pixel_format)
        # print(f"Image format: {pixel_format}, Color type: {is_color}")
        if is_color == 'color':
            # 处理彩色图像格式
            
            # 处理Bayer格式
            if pixel_format == PixelType_Gvsp_BayerGR8:
                img = data.reshape((height, width))
                # 理论上应该转换为BGR，但这个格式似乎有一些问题
                return cv2.cvtColor(img, cv2.COLOR_BayerGR2RGB)
            elif pixel_format == PixelType_Gvsp_BayerRG8:
                img = data.reshape((height, width))
                return cv2.cvtColor(img, cv2.COLOR_BayerRG2BGR)
            elif pixel_format == PixelType_Gvsp_BayerGB8:
                img = data.reshape((height, width))
                return cv2.cvtColor(img, cv2.COLOR_BayerGB2BGR)
            elif pixel_format == PixelType_Gvsp_BayerBG8:
                img = data.reshape((height, width))
                return cv2.cvtColor(img, cv2.COLOR_BayerBG2BGR)
                
            # 处理10位Bayer格式
            elif pixel_format in [PixelType_Gvsp_BayerGR10, PixelType_Gvsp_BayerGR10_Packed]:
                # 注：此处简化处理，实际需根据Packed格式进行解包
                img = data.reshape((height, width))
                # 转换位深度并转为BGR
                img = (img >> 2).astype(np.uint8)  # 10位转8位
                return cv2.cvtColor(img, cv2.COLOR_BayerGR2BGR)
            elif pixel_format in [PixelType_Gvsp_BayerRG10, PixelType_Gvsp_BayerRG10_Packed]:
                img = data.reshape((height, width))
                img = (img >> 2).astype(np.uint8)  # 10位转8位
                return cv2.cvtColor(img, cv2.COLOR_BayerRG2BGR)
            elif pixel_format in [PixelType_Gvsp_BayerGB10, PixelType_Gvsp_BayerGB10_Packed]:
                img = data.reshape((height, width))
                img = (img >> 2).astype(np.uint8)  # 10位转8位
                return cv2.cvtColor(img, cv2.COLOR_BayerGB2BGR)
            elif pixel_format in [PixelType_Gvsp_BayerBG10, PixelType_Gvsp_BayerBG10_Packed]:
                img = data.reshape((height, width))
                img = (img >> 2).astype(np.uint8)  # 10位转8位
                return cv2.cvtColor(img, cv2.COLOR_BayerBG2BGR)
                
            # 处理12位Bayer格式
            elif pixel_format in [PixelType_Gvsp_BayerGR12, PixelType_Gvsp_BayerGR12_Packed]:
                img = data.reshape((height, width))
                img = (img >> 4).astype(np.uint8)  # 12位转8位
                return cv2.cvtColor(img, cv2.COLOR_BayerGR2BGR)
            elif pixel_format in [PixelType_Gvsp_BayerRG12, PixelType_Gvsp_BayerRG12_Packed]:
                img = data.reshape((height, width))
                img = (img >> 4).astype(np.uint8)  # 12位转8位
                return cv2.cvtColor(img, cv2.COLOR_BayerRG2BGR)
            elif pixel_format in [PixelType_Gvsp_BayerGB12, PixelType_Gvsp_BayerGB12_Packed]:
                img = data.reshape((height, width))
                img = (img >> 4).astype(np.uint8)  # 12位转8位
                return cv2.cvtColor(img, cv2.COLOR_BayerGB2BGR)
            elif pixel_format in [PixelType_Gvsp_BayerBG12, PixelType_Gvsp_BayerBG12_Packed]:
                img = data.reshape((height, width))
                img = (img >> 4).astype(np.uint8)  # 12位转8位
                return cv2.cvtColor(img, cv2.COLOR_BayerBG2BGR)
                
            # 处理RGB/BGR格式
            elif pixel_format == PixelType_Gvsp_RGB8_Packed:
                img = data.reshape((height, width, 3))
                return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # 转换为BGR格式
            elif pixel_format == PixelType_Gvsp_BGR8_Packed:
                img = data.reshape((height, width, 3))
                return img  # BGR是OpenCV默认格式，无需转换
            
            # 处理YUV格式
            elif pixel_format == PixelType_Gvsp_YUV422_Packed:
                img = data.reshape((height, int(width * 2)))  # YUV422是2字节一个像素
                return cv2.cvtColor(img, cv2.COLOR_YUV2BGR_YUYV)
            elif pixel_format == PixelType_Gvsp_YUV422_YUYV_Packed:
                img = data.reshape((height, int(width * 2)))  # YUV422是2字节一个像素
                return cv2.cvtColor(img, cv2.COLOR_YUV2BGR_YUYV)
            
            # 未知彩色格式，尝试作为三通道处理
            else:
                try:
                    img = data.reshape((height, width, 3))
                    return img
                except ValueError:
                    print(f"警告：无法处理的彩色格式: {pixel_format}")
                    # 回退到灰度处理
                    img = data.reshape((height, width))
                    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        else:  # 处理单通道灰度图像
            if pixel_format == PixelType_Gvsp_Mono8:
                img = data.reshape((height, width))
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
            # 处理高位深度单通道图像
            elif pixel_format == PixelType_Gvsp_Mono10 or pixel_format == PixelType_Gvsp_Mono10_Packed:
                img = data.reshape((height, width))
                img = (img >> 2).astype(np.uint8)  # 10位转8位
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif pixel_format == PixelType_Gvsp_Mono12 or pixel_format == PixelType_Gvsp_Mono12_Packed:
                img = data.reshape((height, width))
                img = (img >> 4).astype(np.uint8)  # 12位转8位
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                
            # 未知单通道格式，尝试作为灰度图处理
            else:
                try:
                    img = data.reshape((height, width))
                    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                except ValueError:
                    print(f"警告：无法处理的单通道格式: {pixel_format}")
                    # 返回空图像
                    return np.zeros((height, width, 3), dtype=np.uint8)
            
    def get_image(self, debug=False):
        stOutFrame = MV_FRAME_OUT()
        memset(byref(stOutFrame), 0, sizeof(stOutFrame))
        ret = self.cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
        if debug:
            print("case1:get one frame: Width[%d], Height[%d], nFrameNum[%d]" % (
                stOutFrame.stFrameInfo.nWidth, stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nFrameNum))
        
        pData = (c_ubyte * stOutFrame.stFrameInfo.nFrameLen)()
        
        memmove(byref(pData), stOutFrame.pBufAddr, stOutFrame.stFrameInfo.nFrameLen)
        
        data = np.frombuffer(pData, count=int(stOutFrame.stFrameInfo.nFrameLen), dtype=np.uint8)
        
        # 获取图像尺寸和格式
        height = stOutFrame.stFrameInfo.nHeight
        width = stOutFrame.stFrameInfo.nWidth
        pixel_format = stOutFrame.stFrameInfo.enPixelType
        
        # 使用格式转换函数处理图像
        img = self.convert_to_opencv_format(data, height, width, pixel_format)
    
        # 释放图像缓存
        ret =self.cam.MV_CC_FreeImageBuffer(stOutFrame)
        return img

    def close_device(self):
            """
            关闭设备
            """
            ret = self.cam.MV_CC_StopGrabbing()
            if ret != 0:
                print("stop grabbing fail! ret[0x%x]" % ret)
                del self.data_buf
                sys.exit()

            ret = self.cam.MV_CC_CloseDevice()
            if ret != 0:
                print("close device fail! ret[0x%x]" % ret)
                del self.data_buf
                sys.exit()

            ret = self.cam.MV_CC_DestroyHandle()
            if ret != 0:
                print("destroy handle fail! ret[0x%x]" % ret)
                del self.data_buf
                sys.exit()

            del self.data_buf

    def capture_frame(self, name=None, target_dir='./', show=False):
        import os
        image = self.get_image()
        if name is not None:
            os.makedirs(target_dir, exist_ok=True)
            prefix = "capture"
            file_path = os.path.join(target_dir, prefix + name + '.jpg')
            cv2.imwrite(file_path, image)
        if show:
            cv2.namedWindow("image", cv2.WINDOW_NORMAL)
            cv2.imshow("image", image)
            if cv2.waitKey(50) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
        return image
    
    def init(self):
        deviceList = MV_CC_DEVICE_INFO_LIST()
        tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
        self.enum_device(tlayerType, deviceList)
        self.cam, self.data_buf, self.nPayloadSize = self.enable_device(0, deviceList)

if __name__ == "__main__":
    mvs_control = MVSController()

    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    image = mvs_control.get_image()

    h, w = image.shape[:2]
    print("Image shape: %s" % str(image.shape))
    print("Image size: Width[%d], Height[%d]" % (w, h))
    cv2.resizeWindow("image", w//2, h//2)

    while True:
        image = mvs_control.get_image()
        # mvs_control.capture_frame(name="test", target_dir="./", show=True)
        if image is not None:
            print(image.shape)
            height, width = image.shape[:2]
            center_point = (width // 2, height // 2)
            cv2.circle(image, center_point, radius=5, color=(0, 0, 255), thickness=-1)
            cv2.imshow("image", image)
            if cv2.waitKey(50) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

    mvs_control.close_device()