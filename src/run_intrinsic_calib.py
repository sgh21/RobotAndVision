import config.IntrinsicConfig as cfg
from utils.DataCollectorForCalib import DataCollector
from utils.IntrinsicCalibrator import IntrinsicCalibrator

def main():
    print(">>> 启动相机内参全自动标定流程 <<<")
    
    # # === 第一阶段：数据采集 ===
    # collector = DataCollector(cfg)
    # try:
    #     collector.connect_hardware()
    #     collector.execute_collection()
    # except Exception as e:
    #     print(f"[ERROR] 数据采集阶段异常: {e}")
    # finally:
    #     collector.disconnect_hardware()

    # === 第二阶段：内参标定 ===
    print("\n>>> 开始内参核算与误差分析 <<<")
    calibrator = IntrinsicCalibrator(cfg)
    if calibrator.extract_corners():
        calibrator.calibrate_and_evaluate()
        print(f"\n[OK] 标定完成！带角点的重投影图像已保存至: {cfg.OUTPUT_DIR}")
    else:
        print("\n[ERROR] 标定失败：没有足够的有效图像。请检查标定板是否在相机视野内。")

if __name__ == "__main__":
    main()