#!/usr/bin/env python3
"""锅盖热点反射天线 手机支架 STL 生成器 (单位 mm)
依赖: pip install numpy trimesh manifold3d
用法: 改下面的参数后运行  python3 gen_holder.py  -> 在本目录生成 5 个 STL

零件: base(旋钮套筒底座) rod(方杆) slider(滑块) tray(手机托盘) pin(转轴销)
坐标: 玻璃凹面中心 z=0, 杆沿 +z 伸向接收设备。导出的 STL 已按打印姿态摆好。

v2: 锅盖旋钮反装到凹面一侧(梯形圆柱体), 底座改成罩住旋钮的锥形套筒, 热熔胶固定。
"""
import os
import numpy as np
import trimesh
from trimesh import transformations as T

# ---------- 需要你用游标卡尺实测后修改的参数 ----------
# 旋钮是梯形圆柱(圆台): 下面三个数是我按照片估的, 一定要量!
knob_d_bottom = 44    # 旋钮靠近玻璃那一端的直径
knob_d_top    = 40    # 旋钮远离玻璃那一端(顶面)的直径
knob_h        = 26    # 旋钮从玻璃面到顶面的高度
sleeve_wall   = 3     # 套筒壁厚(默认值, 不是实测)
phone_w = 78          # 手机宽度 + 约3mm 余量 (你的手机约 7.5cm)
phone_t = 9           # 手机厚度(去壳, 最厚处), 请量一下
rod_len = 200         # 方杆长度 (杆顶离玻璃 ≈ rod_len + 套筒高度)

# ---------- 一般不用改 ----------
knob_gap = 1.0        # 套筒孔相对旋钮的单边间隙(留余地 + 热熔胶)
bore_extra = 2.0      # 孔比旋钮多深(留余地)
roof_t = 4            # 套筒顶板厚度
rod_w = 12            # 方杆截面(12x12, 比14省料省时)
use_flange = False    # 底座贴玻璃的大圆盘: 关掉更快, 靠套筒边沿+热熔胶固定
tol = 0.20            # 滑动配合单边间隙
tol_tight = 0.08      # 底座插杆(较紧)
flange_t = 4
boss = 24
socket_dep = 25
m3_hole, m3_nut_d, m3_nut_h = 3.2, 6.6, 2.8
sl_s, sl_len = 24, 36
turret_d, turret_h = 22, 12
pin_d = 12
pin_hole_d = pin_d + 0.1
pin_hole_depth = 15
plate_t, rail, rail_h, tray_len = 4.5, 3, 7, 110
cb_d, cb_h = 20.4, 2.2
head_d, head_t = 20, 2
SEG = 64

# 派生尺寸
bore_h = knob_h + bore_extra                         # 套筒孔深
sleeve_od = max(knob_d_bottom, knob_d_top) + 2 * knob_gap + 2 * sleeve_wall
socket_z0 = bore_h + roof_t                             # 方杆底端离玻璃的高度
flange_d = max(64, sleeve_od + 16)

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


def frustum(d_base, d_top, h, base=(0, 0, 0), sections=SEG):
    """圆台: 底面直径 d_base (在 base), 顶面直径 d_top (在 base+h), 沿 +z"""
    a = np.linspace(0, 2 * np.pi, sections, endpoint=False)
    rb, rt = d_base / 2, d_top / 2
    vb = np.c_[rb * np.cos(a), rb * np.sin(a), np.zeros(sections)]
    vt = np.c_[rt * np.cos(a), rt * np.sin(a), np.full(sections, h)]
    verts = np.vstack([vb, vt, [[0, 0, 0], [0, 0, h]]])
    cb, ct = 2 * sections, 2 * sections + 1
    faces = []
    for i in range(sections):
        j = (i + 1) % sections
        faces += [[i, j, sections + j], [i, sections + j, sections + i],
                  [cb, j, i], [ct, sections + i, sections + j]]
    m = trimesh.Trimesh(verts, faces, process=True)
    m.apply_translation(base)
    return m


def union(*ms):
    return trimesh.boolean.union(list(ms), engine="manifold")


def diff(a, *bs):
    return trimesh.boolean.difference([a, *bs], engine="manifold")


def prism(pts_xz, y0, y1):
    """三角棱柱 (筋板): pts_xz 为 XZ 平面三个点, 沿 Y 从 y0 到 y1"""
    v = [[x, y, z] for (x, z) in pts_xz for y in (y0, y1)]
    f = [[0, 2, 4], [1, 5, 3],
         [0, 1, 3], [0, 3, 2],
         [2, 3, 5], [2, 5, 4],
         [4, 5, 1], [4, 1, 0]]
    m = trimesh.Trimesh(vertices=v, faces=f, process=True)
    trimesh.repair.fix_normals(m)
    return m


def prism_xy(pts_xy, z0, z1):
    """三角棱柱: pts_xy 为 XY 平面三个点, 沿 Z 从 z0 到 z1"""
    v = [[x, y, z] for (x, y) in pts_xy for z in (z0, z1)]
    f = [[0, 2, 4], [1, 5, 3], [0, 1, 3], [0, 3, 2], [2, 3, 5], [2, 5, 4], [4, 5, 1], [4, 1, 0]]
    m = trimesh.Trimesh(vertices=v, faces=f, process=True)
    trimesh.repair.fix_normals(m)
    return m


def rotz(m, deg):
    m = m.copy()
    m.apply_transform(T.rotation_matrix(np.radians(deg), [0, 0, 1]))
    return m


# ---------- 零件 ----------
def make_base():
    """旋钮套筒底座: 锥形孔扣在反装的旋钮上, 顶部是方杆插槽"""
    boss_h = socket_z0 + socket_dep
    lock_z = socket_z0 + socket_dep * 0.55
    sk = rod_w + 2 * tol_tight
    g = prism([(boss / 2 - 0.5, socket_z0 - 0.5), (sleeve_od / 2 - 1, socket_z0 - 0.5),
               (boss / 2 - 0.5, socket_z0 + 14)], -1.25, 1.25)
    body = union(
        *([cyl(flange_d, flange_t)] if use_flange else []),
        cyl(sleeve_od, socket_z0),
        box(-boss / 2, -boss / 2, socket_z0 - 1, boss / 2, boss / 2, boss_h),
        *[rotz(g, a) for a in (0, 90, 180, 270)],
    )
    # 锥形孔: 玻璃端直径 = knob_d_bottom, 顶端直径 = knob_d_top (各加间隙)
    bore = frustum(knob_d_bottom + 2 * knob_gap, knob_d_top + 2 * knob_gap,
                   bore_h + 0.01, base=(0, 0, -0.01))
    glue = [rotz(cyl(4, sleeve_od, base=(0, 0, bore_h / 2), axis="x"), a) for a in (0, 120, 240)]
    return diff(
        body,
        bore,
        *glue,                                                                   # 3个注胶孔
        box(-sk / 2, -sk / 2, socket_z0, sk / 2, sk / 2, boss_h + 0.1),         # 插杆方孔
        cyl(m3_hole, boss, base=(0, 0, lock_z), axis="x"),                       # 锁紧螺丝孔
        cyl(m3_nut_d, m3_nut_h + 0.1, base=(boss / 2 - m3_nut_h, 0, lock_z), axis="x", sections=6),
    )


def make_rod():
    """方杆: 刻度 = 杆插到底时该刻度离玻璃的距离(mm); 100/150/200 处是长刻度, 200 处双线(焦点附近)"""
    cuts = []
    d_lo = int(np.ceil((socket_z0 + socket_dep + 5) / 10.0)) * 10
    for D in range(d_lo, int(socket_z0 + rod_len) - 4, 10):
        z = D - socket_z0
        w = rod_w + 1 if D % 50 == 0 else 6
        cuts.append(box(-w / 2, rod_w / 2 - 0.6, z - 0.4, w / 2, rod_w / 2 + 0.5, z + 0.4))
        if D == 200:
            cuts.append(box(-w / 2, rod_w / 2 - 0.6, z + 2.6, w / 2, rod_w / 2 + 0.5, z + 3.4))
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
        box(-sk / 2, -sk / 2, -0.1, sk / 2, sk / 2, sl_len + 0.1),
        cyl(pin_hole_d, pin_hole_depth + 0.1,
            base=(0, sl_s / 2 + turret_h - pin_hole_depth, sl_len / 2), axis="y"),
        cyl(m3_hole, sl_s, base=(0, 0, sl_len / 2), axis="x"),
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
    for sx in (-1, 1):
        for y in (-28, 28):
            cx = sx * (phone_w / 2 + rail + 3.2)
            cuts.append(box(cx - 1.8, y - 10, -0.1, cx + 1.8, y + 10, plate_t + 0.1))
    return diff(union(*parts), *cuts)


def make_pin():
    return union(cyl(head_d, head_t),
                 cyl(pin_d, (plate_t - cb_h) + 12 + 0.5, base=(0, 0, head_t - 0.01)))


def make_cradle():
    """可滑动、可锁紧的手机托架(单个套)。
    一个套在方杆上的滑块(方孔 + M3 锁紧螺丝)直接托在底托下面, 没有立柱。
    充电线孔从底托一直贯穿到滑块底部, 线可以直接穿下去; 杆孔偏在线孔一侧(+X), 避开线孔。
    手机竖放: 背靠后挡板, 底边压在底托上, 两侧挡边 + 前挡唇。
    按使用姿态建模: x 横向, y 向上, z 沿杆朝接收设备; 后挡板贴热床"躺着"打印, 不用支撑。
    读数: 滑块后端面(挡板背面)对着的杆刻度 + 约 10mm = 手机离玻璃的距离。"""
    inner_w = phone_w
    hw = (inner_w + 6) / 2
    slot = phone_t + 2
    back_t, shelf_t, lip_h, plate_h = 4, 5, 6, 42
    hole = rod_w + 0.4                                 # 杆孔(滑动配合)
    bh = hole + 2 * 4.8                                # 滑块高 ~22
    cy = bh / 2                                        # 杆轴线高度
    pk = 8                                             # 充电线孔半宽(16mm 宽)
    x_r = pk + 3.8 + hole / 2                          # 杆孔中心偏到 +X 侧
    bx0 = -(pk + 4)
    bx1 = x_r + hole / 2 + 6.8                         # 右壁 6.8mm: 放 M3 螺母槽
    blen = 32
    y_s0, y_s1 = bh, bh + shelf_t                      # 底托直接坐在滑块上
    front = back_t + slot
    top = y_s1 + plate_h
    parts = [
        box(bx0, 0, 0, bx1, bh + 0.5, blen),                                   # 滑块(套)
        box(-hw, y_s0, 0, hw, y_s1, front + 4),                                # 底托
        box(-hw, y_s0, 0, hw, top, back_t),                                    # 后挡板
        box(-hw, y_s1 - 0.5, front, hw, y_s1 + lip_h, front + 4),              # 前挡唇
    ]
    for sx in (-1, 1):
        wx0 = inner_w / 2 if sx > 0 else -hw
        parts.append(box(wx0, y_s1 - 0.5, 0, wx0 + 3, y_s1 + 40, front))       # 两侧挡边
    cuts = [
        box(x_r - hole / 2, cy - hole / 2, -0.1, x_r + hole / 2, cy + hole / 2, blen + 0.1),   # 方杆通孔
        cyl(m3_hole, bx1 - x_r + 1, base=(x_r, cy, blen / 2), axis="x"),                        # M3 锁紧螺丝孔
        cyl(m3_nut_d, m3_nut_h + 0.1, base=(bx1 - m3_nut_h, cy, blen / 2), axis="x", sections=6),
        box(-pk, -0.1, back_t, pk, y_s1 + 0.1, front),                          # 充电线孔: 贯穿底托和滑块
        box(-26, y_s1 + 14, -0.1, 26, top + 0.1, back_t + 0.1),                 # 减料窗(顶部敞开)
    ]
    return diff(union(*parts), *cuts)


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
    print(f"套筒孔: 玻璃端 d={knob_d_bottom + 2*knob_gap:.1f}, 顶端 d={knob_d_top + 2*knob_gap:.1f}, "
          f"深 {bore_h:.1f}; 套筒外径 {sleeve_od:.1f}; 方杆底端离玻璃 {socket_z0:.1f}; "
          f"杆顶离玻璃 {socket_z0 + rod_len:.0f}")
    save("1_base", make_base())
    save("2_rod", make_rod(), T.rotation_matrix(np.radians(90), [0, 1, 0]))      # 平放打印
    save("3_slider", make_slider(), T.rotation_matrix(np.radians(90), [1, 0, 0]))  # 圆台朝上
    save("4_tray", make_tray())
    save("5_pin", make_pin())
    save("6_cradle", make_cradle())                                              # 后挡板贴热床
