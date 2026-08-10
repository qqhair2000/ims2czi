import os
import h5py
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from imaris_ims_file_reader.ims import ims
from pylibCZIrw import czi
#from pylibCZIrw.czi import Rgb8Color, ChannelDisplaySettingsDataClass, TintingMode

def safe_float_from_attr(attr_val):
    try:
        if isinstance(attr_val, bytes):
            attr_val = attr_val.decode(errors='ignore')
        elif isinstance(attr_val, (list, np.ndarray)):
            if len(attr_val) > 0 and isinstance(attr_val[0], bytes):
                attr_val = b''.join(attr_val).decode(errors='ignore')
            elif len(attr_val) == 1:
                return safe_float_from_attr(attr_val[0])
            else:
                return float('nan')
        return float(attr_val)
    except Exception as e:
        print(f"⚠️ safe_float_from_attr 轉換錯誤 Conversion error: {e}，輸入: {attr_val}")
        return float('nan')

def try_get_pixel_size_from_geometry(h5_path, shape):
    try:
        with h5py.File(h5_path, 'r') as f:
            image_info = f['/DataSetInfo/Image']
#            print("Attributes under /DataSetInfo/Image:")
#            for key in image_info.attrs:
#                print(f"  {key}: {image_info.attrs[key]}")

            ext_max = []
            ext_min = []
            for i in range(3):
                key_max = f"ExtMax{i}"
                key_min = f"ExtMin{i}"
                if key_max in image_info.attrs and key_min in image_info.attrs:
                    max_val = safe_float_from_attr(image_info.attrs[key_max])
                    min_val = safe_float_from_attr(image_info.attrs[key_min])
                    ext_max.append(max_val)
                    ext_min.append(min_val)
                else:
                    print(f"⚠️ {key_max} 或 {key_min} 不存在，使用nan")
                    ext_max.append(float('nan'))
                    ext_min.append(float('nan'))

            if any(np.isnan(ext_max)) or any(np.isnan(ext_min)):
                raise ValueError("ExtMax 或 ExtMin 包含 NaN")

            span = [ext_max[i] - ext_min[i] for i in range(3)]

            _, _, Z, Y, X = shape

            scale_x = span[0] / X * 1e-6
            scale_y = span[1] / Y * 1e-6
            scale_z = span[2] / Z * 1e-6
            return scale_x, scale_y, scale_z
    except Exception as e:
        print(f"⚠️ 讀 geometry 失敗，使用預設值 Failed to read geometry, using default values: {e}")
        return 0.2e-6, 0.2e-6, 1.0e-6

def read_channel_names_from_ims(ims_path):
    channel_names = {}
    try:
        with h5py.File(ims_path, 'r') as f:
            if "/DataSetInfo" in f:
                group = f["/DataSetInfo"]
                for key in group:
                    if key.startswith("Channel"):
                        ch_group = group[key]
                        # 取 Channel 後面的數字，注意是空格還是無空格，例如 Channel 0 或 Channel0
                        parts = key.split()
                        if len(parts) == 2 and parts[0] == "Channel":
                            ch_index = int(parts[1])
                        else:
                            # fallback 如果沒空格直接數字在後
                            ch_index = int(''.join(filter(str.isdigit, key)))
                        name = ch_group.attrs.get("Name", None)
                        if name is not None:
                            if isinstance(name, bytes):
                                name = name.decode(errors="ignore")
                            elif isinstance(name, np.ndarray):
                                # 可能是字元陣列
                                name = name.tobytes().decode(errors="ignore")
                            channel_names[ch_index] = name
                        else:
                            channel_names[ch_index] = f"Ch{ch_index}"
            if not channel_names:
                print("⚠️ 無 Channel metadata，使用預設名稱 No channel metadata found, using default names")
    except Exception as e:
        print(f"❌ 讀取 channel name 失敗 fail: {e}")
    return channel_names

def read_channel_colors_from_ims(ims_path):
    colors = {}
    try:
        with h5py.File(ims_path, 'r') as f:
            group = f.get("/DataSetInfo", {})
            for key in group:
                if key.startswith("Channel"):
                    ch_group = group[key]
                    ch_index = int(''.join(filter(str.isdigit, key)))
                    raw_color = ch_group.attrs.get("Color", None)
                    if raw_color is not None:
                        try:
                            if isinstance(raw_color, bytes):
                                color_str = raw_color.decode(errors='ignore')
                            elif isinstance(raw_color, (np.ndarray, list)):
                                color_str = b''.join(raw_color).decode(errors='ignore') if isinstance(raw_color[0], bytes) else str(raw_color)
                            else:
                                color_str = str(raw_color)

                            rgb_floats = [float(v) for v in color_str.strip().split()]
                            if len(rgb_floats) == 3:
                                r, g, b = [int(round(max(0, min(1, x)) * 255)) for x in rgb_floats]
                                colors[ch_index] = (r, g, b)
                            else:
                                print(f"⚠️ 顏色解析長度錯誤: {color_str}")
                                colors[ch_index] = (0, 0, 0)
                        except Exception as e:
                            print(f"⚠️ 顏色轉換錯誤: {e}, 原始: {raw_color}")
                            colors[ch_index] = (0, 0, 0)
                    else:
                        colors[ch_index] = (0, 0, 0)
    except Exception as e:
        print(f"❌ 讀取 channel color 失敗: {e}")
    return colors




def create_display_settings_from_colors(color_dict):
    settings = {}
    for ch, (r, g, b) in color_dict.items():
        rgb = czi.Rgb8Color(np.uint8(r), np.uint8(g), np.uint8(b))
        setting = czi.ChannelDisplaySettingsDataClass(
            True,
            czi.TintingMode.Color,
            rgb,
            0.0,
            1.0
        )
        settings[ch] = setting
    return settings








def ims_to_czi(ims_path):
    print(f"處理中: {ims_path}")
    a = ims(ims_path)
    T, C, Z, Y, X = a.shape
    out_path = os.path.splitext(ims_path)[0] + ".czi"

    # 讀 channel name
    channel_names = read_channel_names_from_ims(ims_path)
    channel_colors = read_channel_colors_from_ims(ims_path)
    display_settings = create_display_settings_from_colors(channel_colors)
    # 如果沒讀到就用預設
    if not channel_names:
        channel_names = {i: f"Ch{i}" for i in range(C)}
        if C == 2:
            channel_names = {0: "405nm", 1: "488nm"}

    with czi.create_czi(out_path, exist_ok=True) as cz:
        for t in range(T):
            for c in range(C):
                for z in range(Z):
                    
                    # Flip horizontally 
                    
                    plane = np.flip(plane, axis=0)
                    

                    plane = a[t, c, z, :, :]
                    cz.write(data=plane, plane={"C": c, "Z": z, "T": t})

        scale_x, scale_y, scale_z = try_get_pixel_size_from_geometry(ims_path, a.shape)

        # Step 1: 寫入基本 metadata
        cz.write_metadata(
            channel_names=channel_names,
            scale_x=scale_x,
            scale_y=scale_y,
            scale_z=scale_z,
            display_settings=display_settings
        )
        # Step 2: 寫入顏色設定
       # cz.write_metadata(display_settings=display_settings)

    print(f"✅ 轉換完成: {out_path}\n")
    a.close()

def batch_process(folder_path):
    ims_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".ims")]
    if not ims_files:
        messagebox.showinfo("沒有檔案 No files found", "此資料夾中沒有 .ims 檔案")
        return

    for filename in ims_files:
        full_path = os.path.join(folder_path, filename)
        try:
            ims_to_czi(full_path)
        except Exception as e:
            print(f"❌ 轉換失敗: {filename}\n錯誤訊息: {e}\n")

    messagebox.showinfo("完成", "所有 .ims 檔案已轉換為 .czi All .ims files have been converted to .czi")

def select_folder_and_run():
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="選擇含 .ims 的資料夾")
    if folder_path:
        batch_process(folder_path)

if __name__ == "__main__":
    select_folder_and_run()
