import os
from calibrate_helper import Calibrator


def main():
    # CẤU HÌNH
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pic_dir = os.path.join(base_dir, "pic")

    img_calib_dir = os.path.join(pic_dir, "RGB_camera_calib_img")
    img_corners_dir = os.path.join(pic_dir, "RGB_camera_calib_img_corners")
    img_normal_dir = os.path.join(pic_dir, "RGB_camera_normal_img")
    img_normal_undist_dir = os.path.join(pic_dir, "RGB_camera_normal_img_undistorted")

    # Tự động tạo thư mục nếu chưa tồn tại
    for d in [img_calib_dir, img_corners_dir, img_normal_dir, img_normal_undist_dir]:
        os.makedirs(d, exist_ok=True)

    shape_inner = (11, 8)       # Số inner corners (cột, hàng) ← sửa cho đúng checkerboard của bạn
    square_size = 0.017         # mét (17mm)

    # KHỞI TẠO CALIBRATOR
    calibrator = Calibrator(
        img_dir=img_calib_dir,
        shape_inner_corner=shape_inner,
        size_grid=square_size,
        visualization=False      # Đặt True nếu muốn xem ảnh pop-up
    )

    # =============================================
    # CALIBRATE & LƯU ẢNH CORNERS + OXYZ
    # =============================================
    print("\n🚀 BẮT ĐẦU QUÁ TRÌNH CALIBRATION...")
    calibrator.calibrate_camera(save_corners_dir=img_corners_dir)

    # =============================================
    # UNDISTORT ẢNH CALIBRATION (không bắt buộc, nhưng có thể làm)
    # =============================================
    # Nếu bạn muốn lưu ảnh checkerboard đã undistort
    # calibrator.undistort_images(img_calib_dir, "./pic/RGB_camera_calib_img_undistorted")

    # =============================================
    # UNDISTORT ẢNH THƯỜNG
    # =============================================
    if os.path.exists(img_normal_dir) and os.listdir(img_normal_dir):
        calibrator.undistort_images(img_normal_dir, img_normal_undist_dir)
    else:
        print(f"\n⚠ Thư mục {img_normal_dir} trống hoặc không tồn tại. Bỏ qua undistort ảnh thường.")


if __name__ == "__main__":
    main()