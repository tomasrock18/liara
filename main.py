import cv2
import liara  

if __name__ == "__main__":
    
    cam = liara.CameraLab(1, (1920, 1080))

    
    bg = liara.extract_background(cam)

    
    master_contour = liara.extract_master_contour(cam, bg)

    
    calibration_matrix = [
        [1434, 0, 940],
        [0, 1438, 533],
        [0, 0, 1]
    ]

    
    camera_height = 42 - 4.4

    
    rt_matrix = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]

    
    liara.start_detector_tuning(
        cam=cam,
        calibration_matrix=calibration_matrix,
        rt_matrix=rt_matrix,
        bg=bg,
        master_contour=master_contour,
        camera_height=camera_height
    )
