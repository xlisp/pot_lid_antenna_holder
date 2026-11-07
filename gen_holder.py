#!/usr/bin/env python3
"""锅盖热点反射天线 手机支架 STL 生成器 (单位 mm)
依赖: pip install numpy scipy trimesh manifold3d
用法: 改下面的参数后运行  python3 gen_holder.py  -> 在本目录生成 5 个 STL

零件: base(底座,罩住锅盖中心螺丝) rod(方杆) slider(滑块) tray(手机托盘) pin(转轴销)
坐标: 底座贴玻璃的面 z=0, 杆沿 +z 伸向接收设备。导出的 STL 已按打印姿态摆好(无需支撑)。
"""
import os
import numpy as np
import trimesh
from trimesh import transformations as T

# ---------- 需要你实测后修改的参数 ----------
screw_d = 16      # 锅盖凹面中心螺丝头/螺母直径 + 2~3mm 余量
screw_h = 10      # 螺丝/螺母凸出玻璃的高度 + 1mm 余量
phone_w = 80      # 手机宽度(去壳) + 约3mm 余量
rod_len = 210     # 方杆长度 (热床小于 220 就改小, 如 190)

# ---------- 一般不用改 ----------
rod_w = 14
tol = 0.20            # 滑动配合单边间隙
tol_tight = 0.08      # 底座插杆(较紧)
flange_d, flange_t = 64, 4
boss = 28
socket_dep = 35
floor_t = 4
m3_hole, m3_nut_d, m3_nut_h = 3.2, 6.6, 2.8
sl_s, sl_len = 28, 40
turret_d, turret_h = 22, 10
pin_d = 12
pin_hole_d = pin_d + 0.1
pin_hole_depth = 15
plate_t, rail, rail_h, tray_len = 4.5, 3, 7, 110
cb_d, cb_h = 20.4, 2.2
head_d, head_t = 20, 2
SEG = 64

OUT = os.path.dirname(os.path.abspath(__file__))


# ---------- 小工具 ----------
def box(x0, y0, z0, x1, y1, z1):
    m = trimesh.creation.box(extents=[x1 - x0, y1 - y0, z1 - z0])
    m.apply_translation([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2])
    return m


def cyl(d, h, base=(0, 0, 0), axis="z", sections=SEG):
    """圆柱, 从 base 点起沿 axis 方向延伸 h"""
    m = trimesh.creation.cylinder(radius=d / 2, height=h, sections=sections)
    m.apply_translation([0, 0, h / 2])
    if axis == "y":
        m.apply_transform(T.rotation_matrix(np.radians(-90), [1, 0, 0]))
    elif axis == "x":
        m.apply_transform(T.rotation_matrix(np.radians(90), [0, 1, 0]))
    m.apply_translation(base)
    return m


def union(*ms):
    return trimesh.boolean.union(list(ms), engine="manifold")


def diff(a, *bs):
    return trimesh.boolean.difference([a, *bs], engine="manifold")


def prism(pts_xz, y0, y1):
    """三角棱柱 (筋板): pts_xz 为 XZ 平面三个点, 沿 Y 从 y0 到 y1"""
    v = [[x, y, z] for (x, z) in pts_xz for y in (y0, y1)]
    f = [[0, 2, 4], [1, 5, 3],            # 前后三角面
         [0, 1, 3], [0, 3, 2],            # 侧面
         [2, 3, 5], [2, 5, 4],
         [4, 5, 1], [4, 1, 0]]
    m = trimesh.Trimesh(vertices=v, faces=f, process=True)
    trimesh.repair.fix_normals(m)
    return m


def rotz(m, deg):
    m = m.copy()
    m.apply_transform(T.rotation_matrix(np.radians(deg), [0, 0, 1]))
    return m


# ---------- 零件 ----------
def make_base():
    boss_h = screw_h + floor_t + socket_dep
    lock_z = screw_h + floor_t + socket_dep * 0.55
    sk = rod_w + 2 * tol_tight
    g = prism([(boss / 2 - 0.5, 0), (flange_d / 2 - 2, 0), (boss / 2 - 0.5, 22)], -1.25, 1.25)
    body = union(
        cyl(flange_d, flange_t),
        box(-boss / 2, -boss / 2, 0, boss / 2, boss / 2, boss_h),
        *[rotz(g, a) for a in (0, 90, 180, 270)],
    )
    return diff(
        body,
        cyl(screw_d, screw_h + 0.01, base=(0, 0, -0.01)),                       # 罩螺丝的凹槽
        box(-sk / 2, -sk / 2, screw_h + floor_t, sk / 2, sk / 2, boss_h + 0.1),  # 插杆方孔
        cyl(m3_hole, boss, base=(0, 0, lock_z), axis="x"),                       # 锁紧螺丝孔
        cyl(m3_nut_d, m3_nut_h + 0.1, base=(boss / 2 - m3_nut_h, 0, lock_z), axis="x", sections=6),
    )


def make_rod():
    cuts = []
    for z in range(10, int(rod_len) - 4, 10):       # 刻度槽: 每10mm, 每50mm加长
        w = rod_w + 1 if z % 50 == 0 else 6
        cuts.append(box(-w / 2, rod_w / 2 - 0.6, z - 0.4, w / 2, rod_w / 2 + 0.5, z + 0.4))
    r = box(-rod_w / 2, -rod_w / 2, 0, rod_w / 2, rod_w / 2, rod_len)
    return diff(r, *cuts)


def make_slider():
    sk = rod_w + 2 * tol
    body = union(
        box(-sl_s / 2, -sl_s / 2, 0, sl_s / 2, sl_s / 2, sl_len),
        cyl(turret_d, turret_h + 0.01, base=(0, sl_s / 2 - 0.01, sl_len / 2), axis="y"),
    )
    return diff(
        body,
        box(-sk / 2, -sk / 2, -0.1, sk / 2, sk / 2, sl_len + 0.1),                       # 方杆通孔
        cyl(pin_hole_d, pin_hole_depth + 0.1,
            base=(0, sl_s / 2 + turret_h - pin_hole_depth, sl_len / 2), axis="y"),       # 销孔
        cyl(m3_hole, sl_s, base=(0, 0, sl_len / 2), axis="x"),                            # 锁紧螺丝孔
        cyl(m3_nut_d, m3_nut_h + 0.1, base=(sl_s / 2 - m3_nut_h, 0, sl_len / 2), axis="x", sections=6),
    )


def make_tray():
    tray_w = phone_w + 2 * rail + 20
    parts = [box(-tray_w / 2, -tray_len / 2, 0, tray_w / 2, tray_len / 2, plate_t)]
    for sx in (-1, 1):
        x0 = sx * (phone_w / 2 + rail / 2) - rail / 2
        parts.append(box(x0, -tray_len / 2, plate_t, x0 + rail, tray_len / 2, plate_t + rail_h))
    cuts = [cyl(pin_d + 0.6, plate_t + 0.2, base=(0, 0, -0.1)),
            cyl(cb_d, cb_h + 0.1, base=(0, 0, plate_t - cb_h))]
    for sx in (-1, 1):                                  # 绑带孔(橡皮筋/魔术贴)
        for y in (-28, 28):
            cx = sx * (phone_w / 2 + rail + 3.2)
            cuts.append(box(cx - 1.8, y - 10, -0.1, cx + 1.8, y + 10, plate_t + 0.1))
    return diff(union(*parts), *cuts)


def make_pin():
    return union(cyl(head_d, head_t),
                 cyl(pin_d, (plate_t - cb_h) + 12 + 0.5, base=(0, 0, head_t - 0.01)))


def save(name, m, to_print=None):
    if to_print is not None:
        m = m.copy()
        m.apply_transform(to_print)
        m.apply_translation([0, 0, -m.bounds[0][2]])     # 底面落在 z=0
    path = os.path.join(OUT, name + ".stl")
    m.export(path)
    ext = m.bounds[1] - m.bounds[0]
    print(f"{name:8s} watertight={m.is_watertight} vol={m.volume/1000:6.1f}cm3 "
          f"size={ext[0]:.0f}x{ext[1]:.0f}x{ext[2]:.0f}mm")


if __name__ == "__main__":
    save("1_base", make_base())
    save("2_rod", make_rod(), T.rotation_matrix(np.radians(90), [0, 1, 0]))      # 平放打印
    save("3_slider", make_slider(), T.rotation_matrix(np.radians(90), [1, 0, 0]))  # 圆台朝上
    save("4_tray", make_tray())
    save("5_pin", make_pin())
