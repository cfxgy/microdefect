#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""微缺智学——工业缺陷自适应检测系统（Streamlit 界面）。

三个区域：单图检测 / 批量检测 / 历史批次。
检测全部调用 ``src/core/detector.py`` 的真实检测器，不复制任何算法。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.core.batch import (
    DEFAULT_BATCH_ROOT,
    BatchRun,
    create_batch_dir,
    list_batches,
    scan_folder,
)
from src.core.detector import Detector, DetectorError
from src.core.registry import MemoryRegistry, RegistryError, expected_versions
from src.ui import product_pages, services

st.set_page_config(
    page_title="微缺智学——工业缺陷自适应检测系统",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2.0rem; padding-bottom: 2.0rem;}
    .app-title {font-size: 1.9rem; font-weight: 700; margin-bottom: 0.1rem;}
    .app-sub {color: #5b6b7c; margin-bottom: 1.2rem;}
    .badge {display:inline-block; padding:0.35rem 1.0rem; border-radius:0.5rem;
            font-size:1.05rem; font-weight:700; color:#fff;}
    .badge-anomaly {background:#c0392b;}
    .badge-normal {background:#1e8449;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def _options() -> dict[str, list[str]]:
    return services.discover_options()


@st.cache_resource(show_spinner=False)
def _detector(protocol: str, category: str, device: str) -> Detector:
    return Detector(category, protocol=protocol, device=device)


@st.cache_resource(show_spinner=False)
def _config_summary(protocol: str, category: str) -> dict:
    return MemoryRegistry(protocol=protocol).load_config(category).summary()


def _render_result(state: dict) -> None:
    result = state["result"]
    files = state["files"]
    if result.is_anomaly:
        st.markdown('<span class="badge badge-anomaly">检测结论：异常</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-normal">检测结论：正常</span>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("图像级异常分数", f"{result.image_score:.4f}")
    col2.metric("图像级阈值", f"{result.image_threshold:.4f}")
    col3.metric("像素级阈值", f"{result.pixel_threshold:.4f}")
    col4.metric("检测耗时", f"{result.elapsed_time * 1000:.0f} ms")

    st.caption(
        f"类别：{result.category}　协议：{services.PROTOCOL_LABELS.get(result.protocol, result.protocol)}　"
        f"尺寸：{result.image_size[0]}×{result.image_size[1]}　"
        f"特征：{result.model_version['global_feature_version']}"
    )

    img1, img2, img3 = st.columns(3)
    with img1:
        st.image(state["upload_path"], caption="原图", width="stretch")
    with img2:
        st.image(files["heatmap_png"], caption="异常热图", width="stretch")
    with img3:
        st.image(files["overlay_png"], caption="热图叠加", width="stretch")

    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button(
            "下载检测结果 JSON",
            data=Path(files["result_json"]).read_bytes(),
            file_name=Path(files["result_json"]).name,
            mime="application/json",
            key="download_json",
        )
    with d2:
        st.download_button(
            "下载热图 PNG",
            data=Path(files["heatmap_png"]).read_bytes(),
            file_name=Path(files["heatmap_png"]).name,
            mime="image/png",
            key="download_heatmap",
        )
    with d3:
        st.download_button(
            "下载叠加图 PNG",
            data=Path(files["overlay_png"]).read_bytes(),
            file_name=Path(files["overlay_png"]).name,
            mime="image/png",
            key="download_overlay",
        )

    with st.expander("模型与记忆库版本", expanded=False):
        st.json({
            "model_version": result.model_version,
            "memory_bank_version": result.memory_bank_version,
            "scoring_version": result.to_dict()["scoring_version"],
        })


def _render_single_image(options: dict[str, list[str]]) -> None:
    protocols = list(options.keys())
    versions = expected_versions()

    with st.sidebar:
        st.header("单图检测配置")
        protocol = st.selectbox(
            "检测协议",
            options=protocols,
            format_func=lambda p: services.PROTOCOL_LABELS.get(p, p),
            key="protocol",
        )
        categories = options[protocol]
        category = st.selectbox("检测类别", options=categories, key="category")
        device = st.selectbox("推理设备", options=["auto", "cuda", "cpu"], index=0, key="device")

        if category in categories:
            try:
                summary = _config_summary(protocol, category)
                st.caption(
                    f"记忆库：{summary['global_shape']} / {summary['spatial_shape']}　"
                    f"seed={summary['seed']} shot={summary['shot']}"
                )
                st.caption(
                    f"阈值：图像 {summary['image_threshold']:.6f}　像素 {summary['pixel_threshold']:.6f}"
                )
            except RegistryError as exc:
                st.error(f"配置读取失败：{exc}")

        st.divider()
        st.subheader("模型信息")
        st.caption(f"Backbone：{versions['backbone']}")
        st.caption(f"全局特征：{versions['global_feature_version']}")
        st.caption(f"预处理：{versions['preprocess_version']}")
        st.caption("权重：f37072fd…（冻结，不重新训练）")

    left, right = st.columns([1.0, 1.0], gap="large")
    upload_error: str | None = None
    current_signature: str | None = None
    with left:
        st.subheader("1. 图像输入")
        uploaded = st.file_uploader(
            "选择待检测图像（PNG / JPG / JPEG / BMP）",
            type=["png", "jpg", "jpeg", "bmp"],
            key="uploader",
        )
        if uploaded is not None:
            try:
                data_now = uploaded.getvalue()
                services.validate_image_bytes(data_now, uploaded.name)
            except services.ImageValidationError as exc:
                upload_error = str(exc)
                st.error(f"图像不可用：{exc}")
            else:
                current_signature = services.hash_signature(
                    data_now, category, protocol, device
                )
                st.image(uploaded, caption=f"待检测：{uploaded.name}", width="stretch")

    with right:
        st.subheader("2. 开始检测")
        st.write(
            f"当前配置：**{services.PROTOCOL_LABELS.get(protocol, protocol)}** / **{category}** / "
            f"设备 `{device}`"
        )
        st.caption("检测将真实加载冻结模型与 5-shot 记忆库；不会读取测试标签或掩码。")
        # 不使用跨 rerun 的 running 锁：Streamlit 本身按会话串行执行；
        # 任何中断（重新上传、切换 Tab、刷新）都不应把按钮永久锁死。
        run_clicked = st.button(
            "开始检测",
            type="primary",
            disabled=uploaded is None or upload_error is not None,
            key="detect_button",
        )

    if run_clicked and uploaded is not None:
        data = uploaded.getvalue()
        try:
            services.validate_image_bytes(data, uploaded.name)
        except services.ImageValidationError as exc:
            st.error(f"图像不可用：{exc}")
            st.session_state.pop("detection", None)
            st.session_state["detection_error"] = f"图像不可用：{exc}"
            return

        signature = services.hash_signature(data, category, protocol, device)
        cached = st.session_state.get("detection")
        if cached and st.session_state.get("signature") == signature:
            st.info("相同输入与配置的结果已存在，直接复用，未重复执行检测。")
        else:
            try:
                with st.spinner("正在提取特征并计算异常分数…"):
                    engine = _detector(protocol, category, device)
                    state = services.run_detection(
                        data, uploaded.name, category, protocol,
                        device=device, detector=engine,
                    )
                st.session_state["detection"] = state
                st.session_state["signature"] = signature
                st.session_state.pop("detection_error", None)
            except (RegistryError, DetectorError) as exc:
                message = f"检测失败：{exc}"
                st.error(message)
                st.session_state["detection_error"] = message
                st.session_state.pop("detection", None)
            except Exception as exc:  # noqa: BLE001
                message = f"检测异常：{type(exc).__name__}: {exc}"
                st.error(message)
                st.session_state["detection_error"] = message
                st.session_state.pop("detection", None)

    st.divider()
    st.subheader("3. 检测结果")
    state = st.session_state.get("detection")
    error_message = st.session_state.get("detection_error")
    if state is None:
        if error_message:
            st.error(error_message)
        else:
            st.info("请上传图像并点击「开始检测」。")
        return
    if current_signature is None:
        st.info("当前没有有效的待检测图像，请重新上传。")
        return
    if state.get("signature") != current_signature:
        st.info("已选择新的图像或配置，请点击「开始检测」获取最新结果。")
        return
    if Path(state["upload_path"]).exists() and Path(state["files"]["heatmap_png"]).exists():
        _render_result(state)
    else:
        st.error("结果文件缺失，请重新检测。")


def _batch_table(rows: list[dict], batch_dir: Path | None, key_prefix: str) -> None:
    if not rows:
        st.info("暂无记录。")
        return
    filter_option = st.selectbox(
        "筛选", ["全部", "正常", "异常", "检测失败", "已停止"], key=f"{key_prefix}_filter"
    )
    selection = {
        "正常": lambda r: r["status"] == "success" and str(r["is_anomaly"]) == "0",
        "异常": lambda r: r["status"] == "success" and str(r["is_anomaly"]) == "1",
        "检测失败": lambda r: r["status"] == "failed",
        "已停止": lambda r: r["status"] == "stopped",
    }
    filtered = rows if filter_option == "全部" else [r for r in rows if selection[filter_option](r)]
    frame = pd.DataFrame([
        {
            "序号": row["index"],
            "文件名": row["file_name"],
            "类别": row["category"],
            "图像分数": row["image_score"],
            "图像阈值": row["image_threshold"],
            "结论": ("异常" if str(row["is_anomaly"]) == "1"
                     else "正常" if str(row["is_anomaly"]) == "0" else ""),
            "耗时(s)": row["elapsed_time"],
            "状态": {"success": "成功", "failed": "失败", "stopped": "已停止"}.get(row["status"], row["status"]),
            "错误": row["error"],
            "热图": row["heatmap_png"],
            "叠加图": row["overlay_png"],
        }
        for row in filtered
    ])
    st.dataframe(frame, width="stretch", hide_index=True)

    if not filtered:
        st.warning("当前筛选条件下没有记录。")
        return
    labels = [f"#{row['index']} {row['file_name']}（{row['status']}）" for row in filtered]
    chosen = st.selectbox("查看单张图片", labels, key=f"{key_prefix}_view")
    row = filtered[labels.index(chosen)]
    if batch_dir is not None and row["status"] == "success":
        c1, c2, c3 = st.columns(3)
        with c1:
            if Path(row["image_path"]).is_file():
                st.image(row["image_path"], caption="原图", width="stretch")
            else:
                st.warning(f"原图已不存在：{row['image_path']}")
        with c2:
            st.image(str(batch_dir / row["heatmap_png"]), caption="异常热图", width="stretch")
        with c3:
            st.image(str(batch_dir / row["overlay_png"]), caption="热图叠加", width="stretch")
        with st.expander("单张检测 JSON", expanded=False):
            st.json(Path(batch_dir / row["result_json"]).read_text(encoding="utf-8"))


def _render_batch(options: dict[str, list[str]]) -> None:
    st.subheader("批量检测")
    st.caption("扫描本地文件夹，逐张调用已验收的单图检测器；失败图片不会中断整批。")

    folder = st.text_input("图像文件夹路径", key="batch_folder",
                           placeholder=r"例如 D:\data\inspection\2026-11-11")
    out_root = st.text_input("结果保存根目录", value=str(DEFAULT_BATCH_ROOT), key="batch_out_root")
    recursive = st.checkbox("递归扫描子文件夹", value=True, key="batch_recursive")

    c1, c2, c3 = st.columns(3)
    with c1:
        protocol = st.selectbox(
            "批量协议", options=list(options.keys()),
            format_func=lambda p: services.PROTOCOL_LABELS.get(p, p), key="batch_protocol",
        )
    with c2:
        category = st.selectbox("批量类别", options=options[protocol], key="batch_category")
    with c3:
        device = st.selectbox("推理设备 ", options=["auto", "cuda", "cpu"], index=0, key="batch_device")

    try:
        batch_cfg = _config_summary(protocol, category)
        batch_versions = expected_versions()
        st.caption(
            f"模型：{batch_versions['backbone']}（{batch_versions['global_feature_version']}）　"
            f"记忆库：{batch_cfg['global_shape']} / {batch_cfg['spatial_shape']}　"
            f"seed={batch_cfg['seed']} shot={batch_cfg['shot']}　"
            f"阈值：图像 {batch_cfg['image_threshold']:.6f} / 像素 {batch_cfg['pixel_threshold']:.6f}"
        )
    except RegistryError as exc:
        st.error(f"批量配置不可用：{exc}")

    if st.button("扫描文件夹", key="batch_scan_button", disabled=not folder.strip()):
        try:
            scan = scan_folder(folder.strip(), recursive=recursive)
        except (NotADirectoryError, OSError) as exc:
            st.error(f"扫描失败：{exc}")
            st.session_state.pop("batch_scan", None)
        else:
            st.session_state["batch_scan"] = {
                **scan.summary(),
                "valid_paths": [str(p) for p in scan.valid],
            }
            st.session_state.pop("batch_run", None)

    scan_state = st.session_state.get("batch_scan")
    if scan_state:
        m1, m2, m3 = st.columns(3)
        m1.metric("有效图片", scan_state["valid"])
        m2.metric("无效/不支持", scan_state["invalid"])
        m3.metric("文件总数", scan_state["total_files"])
        st.caption(f"扫描目录：{scan_state['folder']}　递归：{scan_state['recursive']}")
        st.caption(f"预计检测 {scan_state['valid']} 张图片。")
        if scan_state["invalid"]:
            with st.expander(f"无效文件（{scan_state['invalid']}）", expanded=False):
                st.dataframe(pd.DataFrame(scan_state["invalid_items"]), width="stretch", hide_index=True)
        if scan_state["valid"] == 0:
            st.warning("该文件夹没有可检测的图片，请更换目录。")

    run = st.session_state.get("batch_run")
    running = run is not None and not run.finished
    if st.button(
        "开始批量检测",
        type="primary",
        key="batch_start_button",
        disabled=(not scan_state) or scan_state["valid"] == 0 or running,
    ):
        scan_state = st.session_state["batch_scan"]
        try:
            engine = _detector(protocol, category, device)
            batch_dir = create_batch_dir(out_root)
        except (RegistryError, DetectorError, OSError) as exc:
            st.error(f"无法启动批量检测：{exc}")
        else:
            st.session_state["batch_run"] = BatchRun(
                [Path(p) for p in scan_state["valid_paths"]], category, protocol,
                batch_dir, engine, device=device,
            )
            st.rerun()

    run = st.session_state.get("batch_run")
    if run is not None and not run.finished:
        if st.button("停止后续任务", key="batch_stop_button"):
            run.request_stop()
        progress = (run.index / run.total) if run.total else 0.0
        st.progress(min(progress, 1.0), text=f"进度 {run.index}/{run.total}")
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("成功", run.success)
        p2.metric("失败", run.failed)
        p3.metric("正常", run.normal)
        p4.metric("异常", run.anomaly)
        run.step()
        if not run.finished:
            st.rerun()

    run = st.session_state.get("batch_run")
    if run is not None and run.finished:
        summary = run.summary()
        st.success(f"批次完成：{summary['batch_id']}")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("总图片", summary["total"])
        s2.metric("成功", summary["success"])
        s3.metric("失败", summary["failed"])
        s4.metric("已停止", summary["stopped"])
        s5, s6, s7, s8 = st.columns(4)
        s5.metric("正常", summary["normal"])
        s6.metric("异常", summary["anomaly"])
        s7.metric("总耗时(s)", summary["elapsed_seconds"])
        s8.metric("开始", summary["started_at"][-8:])
        st.caption(f"批次目录：{summary['batch_dir']}")
        _batch_table(run.rows, run.batch_dir, "batch_current")
    elif run is None:
        st.info("扫描文件夹后即可开始批量检测。")


def _render_history() -> None:
    st.subheader("历史批次")
    root = st.text_input("批次根目录", value=str(DEFAULT_BATCH_ROOT), key="history_root")
    batches = list_batches(root)
    if not batches:
        st.info("该目录下暂无历史批次。")
        return
    frame = pd.DataFrame([
        {
            "批次": item["batch_id"],
            "协议": item.get("protocol", ""),
            "类别": item.get("category", ""),
            "总数": item.get("total", 0),
            "成功": item.get("success", 0),
            "失败": item.get("failed", 0),
            "正常": item.get("normal", 0),
            "异常": item.get("anomaly", 0),
            "开始时间": item.get("started_at", ""),
            "结束时间": item.get("finished_at", ""),
        }
        for item in batches
    ])
    st.dataframe(frame, width="stretch", hide_index=True)
    labels = [f"{item['batch_id']}（{item.get('category', '')}）" for item in batches]
    chosen = st.selectbox("查看批次", labels, key="history_batch")
    summary = batches[labels.index(chosen)]
    if summary.get("finished") is False:
        st.warning("该批次尚未正常结束（可能被中断）。")
    _batch_table(summary.get("images", []), Path(summary["batch_dir"]), "history")


def main() -> None:
    st.markdown('<div class="app-title">微缺智学——工业缺陷自适应检测系统</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-sub">上传图像或扫描文件夹，使用冻结 ResNet18 特征与正常记忆库'
        '完成图像级判定与像素级缺陷定位。</div>',
        unsafe_allow_html=True,
    )

    options = _options()
    product_root = services.product_root()
    tab_single, tab_batch, tab_history, tab_register, tab_product, tab_product_batch, tab_product_history, tab_drift, tab_feedback = st.tabs(
        ["单图检测", "批量检测", "历史批次", "产品注册", "产品检测", "产品批量", "产品批次历史", "产品漂移", "人工审核"]
    )
    with tab_single:
        if options:
            _render_single_image(options)
        else:
            st.error("未发现可用的 MVTec 记忆库配置；产品级检测请使用「产品检测」页。")
    with tab_batch:
        if options:
            _render_batch(options)
        else:
            st.info("没有可用的 MVTec 记忆库配置，批量检测不可用。")
    with tab_history:
        _render_history()
    with tab_register:
        product_pages.render_registration_tab(product_root)
    with tab_product:
        product_pages.render_detection_tab(product_root)
    with tab_product_batch:
        product_pages.render_batch_tab(product_root)
    with tab_product_history:
        product_pages.render_batch_history_tab(product_root)
    with tab_drift:
        product_pages.render_drift_tab(product_root)
    with tab_feedback:
        product_pages.render_feedback_tab(product_root)


main()
