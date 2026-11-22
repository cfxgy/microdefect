#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P3.2 产品注册 / 选择 / 单图检测页面。

页面层只负责交互与展示；注册调用 P2.1、检测调用 P3.1.1，不复制算法。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st

from src.core.product_detector import ProductDetectorError
from src.core.products import ProductStoreError
from src.ui import services

PRODUCT_TYPES = ["png", "jpg", "jpeg", "bmp"]


def _upload_items(uploaded) -> list[tuple[str, bytes]]:
    if not uploaded:
        return []
    if isinstance(uploaded, list):
        return [(item.name, item.getvalue()) for item in uploaded]
    return [(uploaded.name, uploaded.getvalue())]


def _registration_fingerprint(name: str, category: str, calibration_count: int,
                              refs, cals, defects, tests_normal, tests_defect) -> str:
    digest = hashlib.sha256()
    digest.update(str(name).encode("utf-8"))
    digest.update(b"\0" + str(category).encode("utf-8"))
    digest.update(b"\0" + str(calibration_count).encode("ascii"))
    for role, items in (
        ("reference", refs), ("calibration", cals), ("defect", defects),
        ("test_normal", tests_normal), ("test_defect", tests_defect),
    ):
        digest.update(b"\0" + role.encode("ascii"))
        ordered = sorted(items, key=lambda item: (item[0], hashlib.sha256(item[1]).hexdigest()))
        for filename, data in ordered:
            digest.update(b"\0" + filename.encode("utf-8"))
            digest.update(b"\0" + hashlib.sha256(data).digest())
    return digest.hexdigest()


def _show_registration_result(result: dict, root: Path) -> None:
    summary = result["summary"]
    st.success(f"产品已注册并激活：{summary['name']}")
    st.caption(f"产品根目录：{root}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("产品 ID", summary["product_id"][:12])
    c2.metric("当前版本", f"v{summary['current_version']}")
    c3.metric("参考图片", summary["n_reference"])
    c4.metric("校准图片", summary["n_calibration"])
    st.json({
        "product_id": summary["product_id"],
        "name": summary["name"],
        "status": summary["status"],
        "current_version": summary["current_version"],
        "version": summary["version"],
        "manifest_sha256": summary["manifest_sha256"],
        "image_threshold": summary["image_threshold"],
        "pixel_threshold": summary["pixel_threshold"],
        "validation_ok": result["validation"].get("ok"),
        "source_status": result.get("source_status", "legacy"),
        "low_sample_warning": result["low_sample_warning"],
        "independent_test_provided": result["independent_test_provided"],
        "independent_test_validated": False,
    })
    if result["low_sample_warning"]:
        st.warning("校准样本为 20–24 张，数量偏少，阈值波动风险较高。")
    st.warning(
        "校准阈值是有限正常样本上的经验分位数，不是 conformal 校准，"
        "不提供固定 FPR 保证。"
    )
    if summary.get("independent_test_provided"):
        st.caption("已登记独立测试集，但尚未完成独立性能验收（independent_test_validated=false）。")


def render_registration_tab(root: Path) -> None:
    st.subheader("产品注册")
    st.caption(
        "使用 P2.1 正式注册流程：正常参考图 → 冻结特征记忆库 → 独立校准阈值。"
        "注册成功后，上传的参考、校准及其他样本图片会保存在本地产品版本目录中，"
        "用于来源追溯和后续维护。"
        f"正式产品根目录：{root}"
    )
    name = st.text_input("产品名称", key="product_name", placeholder="例如：垫片-A")
    category = st.text_input("产品类别（可选）", key="product_category", placeholder="例如：washer")
    calibration_count = st.number_input(
        "校准图片数量（20–200，默认 25）",
        min_value=20, max_value=200, value=25, step=1,
        key="product_calibration_count",
    )
    reference_uploaded = st.file_uploader(
        "正常参考图片（至少 5 张）", type=PRODUCT_TYPES,
        accept_multiple_files=True, key="product_reference_uploader",
    )
    calibration_uploaded = st.file_uploader(
        "独立正常校准图片（不能与参考图重复）", type=PRODUCT_TYPES,
        accept_multiple_files=True, key="product_calibration_uploader",
    )
    with st.expander("可选：缺陷图 / 独立测试图", expanded=False):
        defect_uploaded = st.file_uploader(
            "缺陷图片（可选，仅登记，不参与评分）", type=PRODUCT_TYPES,
            accept_multiple_files=True, key="product_defect_uploader",
        )
        test_normal_uploaded = st.file_uploader(
            "独立测试正常图片（可选）", type=PRODUCT_TYPES,
            accept_multiple_files=True, key="product_test_normal_uploader",
        )
        test_defect_uploaded = st.file_uploader(
            "独立测试异常图片（可选）", type=PRODUCT_TYPES,
            accept_multiple_files=True, key="product_test_defect_uploader",
        )

    refs = _upload_items(reference_uploaded)
    cals = _upload_items(calibration_uploaded)
    defects = _upload_items(defect_uploaded)
    tests_normal = _upload_items(test_normal_uploaded)
    tests_defect = _upload_items(test_defect_uploaded)
    try:
        fingerprint, _ = services.registration_fingerprint_for_uploads(
            name=name, category=category,
            calibration_count=int(calibration_count),
            reference_files=refs, calibration_files=cals,
            defect_files=defects,
            test_normal_files=tests_normal,
            test_defect_files=tests_defect,
        )
    except Exception:  # noqa: BLE001 - 权重缺失等由提交阶段明确报错
        fingerprint = _registration_fingerprint(
            name, category, int(calibration_count),
            refs, cals, defects, tests_normal, tests_defect,
        )
    duplicate_state = None
    unavailable = []
    if len(refs) >= 5 and len(cals) == int(calibration_count):
        try:
            duplicate_state = services.find_duplicate_registration(
                name=name, category=category,
                calibration_count=int(calibration_count),
                reference_files=refs, calibration_files=cals,
                defect_files=defects,
                test_normal_files=tests_normal,
                test_defect_files=tests_defect,
                root=root,
            )
            unavailable = duplicate_state.get("unavailable", [])
        except Exception as exc:  # noqa: BLE001 - 权重缺失等由提交阶段明确报错
            duplicate_state = None
            unavailable = [{
                "product_id": "-",
                "version": 0,
                "reason": f"判重检查失败：{type(exc).__name__}: {exc}",
            }]
    duplicate = bool(duplicate_state and duplicate_state.get("duplicate"))
    matched = duplicate_state.get("matched") if duplicate_state else None
    already_registered = st.session_state.get("product_registered_fingerprint") == fingerprint
    if duplicate:
        already_registered = True
    st.caption(f"当前：参考 {len(refs)} 张；校准 {len(cals)} 张（设置 {calibration_count} 张）。")
    form_error = None
    if len(refs) < 5:
        form_error = "正常参考图片至少需要 5 张。"
    elif len(cals) != int(calibration_count):
        form_error = f"校准图片实际 {len(cals)} 张，与设置 {calibration_count} 张不一致。"
    elif 20 <= len(cals) <= 24:
        st.warning("校准样本仅 20–24 张，数量偏少；仍可注册，但阈值不确定性较高。")
    if form_error:
        st.error(form_error)
    if duplicate and matched:
        st.warning(
            "已存在相同注册内容："
            f"product_id={matched['product_id']}，version={matched['version']}，"
            f"manifest={str(matched.get('manifest_sha256', ''))[:12]}…；"
            "不会创建重复产品。修改名称/图片/分组后可注册新产品。"
        )
    if unavailable:
        details = "；".join(
            f"{item.get('product_id')}/v{item.get('version')}: {item.get('reason')}"
            for item in unavailable[:5]
        )
        st.warning(
            "存在旧产品无法可靠判重（不会阻止本次注册，但不能宣称已保证全局无重复）："
            + details
        )
    if already_registered and not duplicate:
        st.info("当前表单已成功注册；为避免重复创建产品，注册按钮已锁定。修改名称、图片或数量后可注册另一个产品。")

    clicked = st.button(
        "注册产品", type="primary", key="product_register_button",
        disabled=bool(form_error) or already_registered,
    )
    if clicked:
        try:
            with st.spinner("正在提取参考特征、计算校准阈值并提交产品…"):
                result = services.register_product_from_uploads(
                    name=name, category=category,
                    reference_files=refs,
                    calibration_files=cals,
                    calibration_count=int(calibration_count),
                    defect_files=defects,
                    test_normal_files=tests_normal,
                    test_defect_files=tests_defect,
                    root=root, device="auto",
                )
            if result.get("duplicate"):
                st.session_state["product_duplicate"] = result["existing"]
                st.session_state["product_registered_fingerprint"] = fingerprint
                st.session_state.pop("product_register_result", None)
                st.session_state.pop("product_register_error", None)
                st.rerun()
            st.session_state["product_register_result"] = result
            st.session_state["product_registered_fingerprint"] = fingerprint
            st.session_state.pop("product_duplicate", None)
            st.session_state.pop("product_register_error", None)
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.session_state["product_register_error"] = f"{type(exc).__name__}: {exc}"
            st.session_state.pop("product_register_result", None)

    error = st.session_state.get("product_register_error")
    if error:
        st.error(f"注册失败：{error}")
    result = st.session_state.get("product_register_result")
    if result:
        _show_registration_result(result, root)


def _product_table(cards: list[dict]) -> None:
    frame = pd.DataFrame([
        {
            "产品名称": card["name"],
            "product_id": card["product_id"],
            "状态": card["status"],
            "current_version": card["current_version"],
            "版本状态": card["version_status"],
            "注册时间": card["created_at"],
            "来源样本": card.get("source_status", "legacy"),
            "完整性": "通过" if card["validation_ok"] else "失败",
            "可检测": "是" if card["detectable"] else "否",
            "不可检测原因": card["reason"],
        }
        for card in cards
    ])
    st.dataframe(frame, width="stretch", hide_index=True)


def _show_detection_result(state: dict) -> None:
    result = state["result"]
    files = state["files"]
    if result.is_anomaly:
        st.markdown('<span class="badge badge-anomaly">检测结论：异常</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-normal">检测结论：正常</span>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("图像级异常分数", f"{result.image_score:.6f}")
    c2.metric("图像级阈值", f"{result.image_threshold:.6f}")
    c3.metric("推理耗时", f"{result.inference_time:.4f} s")
    c4.metric("当前版本", f"v{result.version}")

    st.caption(
        f"产品：{result.product_name}（{result.product_id}）　"
        f"input_image_sha256：{result.input_image_sha256}"
    )
    st.caption(
        f"bank_version：{result.bank_version}　manifest：{result.manifest_sha256}"
    )
    st.caption(
        f"特征：{result.feature_extractor_version}　权重：{result.weights_sha256[:16]}…　"
        f"pixel_threshold：{result.pixel_threshold:.6f}"
    )

    img1, img2, img3, img4 = st.columns(4)
    with img1:
        st.image(state["upload_path"], caption="原始输入", width="stretch")
    with img2:
        st.image(files["heatmap_png"], caption="异常热图", width="stretch")
    with img3:
        st.image(files["overlay_png"], caption="热图叠加", width="stretch")
    with img4:
        if files.get("anomaly_mask_png") and Path(files["anomaly_mask_png"]).is_file():
            st.image(files["anomaly_mask_png"], caption="anomaly mask（阈值参考）", width="stretch")
        else:
            st.info("本次未生成 mask。")

    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.download_button(
            "下载结果 JSON", data=Path(files["result_json"]).read_bytes(),
            file_name=Path(files["result_json"]).name,
            mime="application/json", key="product_download_json",
        )
    with d2:
        st.download_button(
            "下载热图 PNG", data=Path(files["heatmap_png"]).read_bytes(),
            file_name=Path(files["heatmap_png"]).name,
            mime="image/png", key="product_download_heatmap",
        )
    with d3:
        st.download_button(
            "下载叠加图 PNG", data=Path(files["overlay_png"]).read_bytes(),
            file_name=Path(files["overlay_png"]).name,
            mime="image/png", key="product_download_overlay",
        )
    with d4:
        if files.get("anomaly_mask_png") and Path(files["anomaly_mask_png"]).is_file():
            st.download_button(
                "下载 mask PNG", data=Path(files["anomaly_mask_png"]).read_bytes(),
                file_name=Path(files["anomaly_mask_png"]).name,
                mime="image/png", key="product_download_mask",
            )

    st.warning(
        "当前正式判定仅为图片级正常/异常；像素 mask 只是 pixel_threshold 参考，"
        "不代表已经通过像素级定位性能验收。"
    )
    if st.button("提交人工审核", key=f"submit_feedback_{result.image_sha256[:12]}"):
        try:
            feedback = services.create_product_feedback(
                result.product_id,
                product_version=result.version,
                source_type="detection",
                source_id=f"detection:{result.image_sha256}",
                image_path=state["upload_path"],
                image_score=result.image_score,
                image_threshold=result.image_threshold,
                image_status=result.image_status,
                original_filename=Path(result.image_path).name,
            )
            st.success(f"已提交人工审核：{feedback['feedback_id']}")
        except Exception as exc:  # noqa: BLE001
            st.error(f"提交审核失败：{exc}")
    with st.expander("完整检测 JSON", expanded=False):
        st.json(result.to_dict())


def _batch_items_frame(payload: dict) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "序号": item.get("index"),
            "原始文件名": item.get("original_filename"),
            "保存文件名": item.get("stored_filename"),
            "输入SHA256": (item.get("input_sha256") or "")[:16] + "…",
            "状态": item.get("status"),
            "分数": item.get("image_score"),
            "阈值": item.get("image_threshold"),
            "结论": ({"success": ("异常" if item.get("is_anomaly") else "正常"),
                     "failed": "失败", "unprocessed": "未处理"}.get(item.get("status"))),
            "耗时(s)": item.get("inference_time"),
            "错误": item.get("error"),
        }
        for item in payload.get("items") or []
    ])


def _render_batch_detail(payload: dict) -> None:
    st.caption(
        f"产品：{payload.get('product_name')}（{payload.get('product_id')}）　"
        f"版本：v{payload.get('product_version')}　"
        f"manifest：{str(payload.get('manifest_sha256', ''))[:16]}…　"
        f"bank：{str(payload.get('bank_version', ''))[:32]}…"
    )
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("总数", payload.get("total", 0))
    c2.metric("成功", payload.get("success", 0))
    c3.metric("失败", payload.get("failed", 0))
    c4.metric("未处理", payload.get("unprocessed", 0))
    c5.metric("状态", payload.get("status", ""))
    st.caption(
        f"正常 {payload.get('normal', 0)}　异常 {payload.get('anomaly', 0)}　"
        f"开始 {payload.get('created_at', '')}　结束 {payload.get('finished_at') or '—'}"
    )
    st.dataframe(_batch_items_frame(payload), width="stretch", hide_index=True)
    batch_id = payload["batch_id"]
    successes = [item for item in payload.get("items") or []
                 if item.get("status") == "success" and item.get("files")]
    if successes:
        labels = [f"#{item['index']} {item['original_filename']}" for item in successes]
        chosen = st.selectbox("查看单张批次结果", labels,
                              key=f"batch_item_{batch_id}")
        item = successes[labels.index(chosen)]
        files = item.get("files") or {}
        try:
            images = st.columns(4)
            for column, key, caption in (
                (images[0], "input_path", "原图"),
                (images[1], "heatmap_png", "异常热图"),
                (images[2], "overlay_png", "热图叠加"),
                (images[3], "anomaly_mask_png", "anomaly mask"),
            ):
                if files.get(key):
                    data = services.product_batch_file_bytes(batch_id, files[key])
                    with column:
                        st.image(data, caption=caption, width="stretch")
            downloads = [
                ("热图", "heatmap_png", "image/png"),
                ("叠加图", "overlay_png", "image/png"),
                ("mask", "anomaly_mask_png", "image/png"),
                ("结果JSON", "result_json", "application/json"),
            ]
            cols = st.columns(len(downloads))
            for column, (label, key, mime) in zip(cols, downloads):
                if files.get(key):
                    with column:
                        st.download_button(
                            f"下载{label}",
                            data=services.product_batch_file_bytes(batch_id, files[key]),
                            file_name=Path(files[key]).name, mime=mime,
                            key=f"batch_item_dl_{batch_id}_{item['index']}_{key}",
                        )
        except Exception as exc:  # noqa: BLE001
            st.warning(f"单张结果文件不可读：{exc}")
        if st.button("提交该图人工审核",
                     key=f"batch_feedback_{batch_id}_{item['index']}"):
            try:
                feedback = services.create_product_feedback_from_batch(
                    payload["product_id"], product_version=payload["product_version"],
                    batch_id=batch_id, item=item,
                )
                st.success(f"已创建待审核反馈：{feedback['feedback_id']}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"创建反馈失败：{exc}")

    downloads = []
    try:
        downloads.append(("CSV 汇总", services.product_batch_file_bytes(
            batch_id, "results.csv"), "text/csv", "results.csv"))
        downloads.append(("JSON 完整记录", services.product_batch_file_bytes(
            batch_id, "batch.json"), "application/json", "batch.json"))
        downloads.append(("ZIP 打包", services.product_batch_zip(batch_id).read_bytes(),
                          "application/zip", f"{batch_id}.zip"))
    except Exception as exc:  # noqa: BLE001
        st.warning(f"部分下载不可用：{exc}")
    cols = st.columns(max(1, len(downloads)))
    for column, (label, data, mime, filename) in zip(cols, downloads):
        with column:
            st.download_button(label, data=data, file_name=filename, mime=mime,
                               key=f"batch_dl_{batch_id}_{label}")
    st.warning("像素 mask 仅为 pixel_threshold 参考，不代表已验证的精确缺陷定位。")


def render_batch_tab(root: Path) -> None:
    st.subheader("产品批量检测")
    st.caption("一个批次固定一个产品 current_version；逐张检测，单图失败不影响其他图片。")
    try:
        cards = services.list_product_cards(root)
    except Exception as exc:  # noqa: BLE001
        st.error(f"产品列表读取失败：{exc}")
        return
    detectable = [card for card in cards if card["detectable"]]
    if not detectable:
        st.warning("当前没有可检测产品。")
        return
    labels = [f"{card['name']} · {card['product_id'][:8]} · v{card['current_version']}"
              for card in detectable]
    selected = st.selectbox("选择批量产品", labels, key="product_batch_select")
    card = detectable[labels.index(selected)]
    uploaded = st.file_uploader(
        "批量图片（可多选；重名文件会独立保存）", type=PRODUCT_TYPES,
        accept_multiple_files=True, key="product_batch_uploader",
    )
    uploads = _upload_items(uploaded)
    device = st.selectbox("批量推理设备", ["auto", "cuda", "cpu"], index=0,
                          key="product_batch_device")
    st.caption(f"已选择 {len(uploads)} 张图片；批次目录：cache/product_batches")
    if st.button("开始产品批次", type="primary", key="product_batch_start",
                 disabled=not uploads):
        try:
            with st.spinner("正在校验产品并保存批次输入…"):
                run = services.create_product_batch(
                    card["product_id"], uploads, root=root, device=device,
                )
            st.session_state["product_batch_run"] = run
            st.session_state.pop("product_batch_error", None)
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.session_state["product_batch_error"] = f"{type(exc).__name__}: {exc}"
            st.session_state.pop("product_batch_run", None)

    error = st.session_state.get("product_batch_error")
    if error:
        st.error(f"批量检测启动失败：{error}")
    run = st.session_state.get("product_batch_run")
    if run is None:
        st.info("选择产品与图片后开始批量检测。")
        return
    if run.product_id != card["product_id"]:
        st.info("当前选择的产品与已有批次不同，请开始新批次。")
        return
    if not run.finished:
        st.progress(min(run.index / max(run.total, 1), 1.0),
                    text=f"进度 {run.index}/{run.total}")
        if st.button("停止后续检测", key="product_batch_stop"):
            run.request_stop()
        try:
            run.step()
        except Exception as exc:  # noqa: BLE001
            st.session_state["product_batch_error"] = f"批次写入失败：{type(exc).__name__}: {exc}"
            st.error(st.session_state["product_batch_error"])
            return
        if not run.finished:
            st.rerun()
    payload = run.to_dict()
    _render_batch_detail(payload)


def render_batch_history_tab(root: Path) -> None:
    st.subheader("产品批次历史")
    st.caption("历史记录只读；批次始终显示其实际 product_version，不随 current_version 改写。")
    try:
        batches = services.list_product_batches(root=None)
    except Exception as exc:  # noqa: BLE001
        st.error(f"批次历史读取失败：{exc}")
        return
    if not batches:
        st.info("暂无产品批次记录。")
        return
    product_options = sorted({(b["product_id"], b.get("product_name", "")) for b in batches})
    labels = ["全部产品"] + [f"{name} · {pid[:8]}" for pid, name in product_options]
    chosen = st.selectbox("按产品筛选", labels, key="product_batch_history_product")
    selected_product = None if chosen == "全部产品" else product_options[labels.index(chosen) - 1][0]
    filtered = [b for b in batches
                if selected_product is None or b["product_id"] == selected_product]
    frame = pd.DataFrame([
        {
            "批次": b["batch_id"],
            "产品": b.get("product_name"),
            "产品ID": b.get("product_id"),
            "版本": b.get("product_version"),
            "状态": b.get("status"),
            "总数": b.get("total"),
            "正常": b.get("normal"),
            "异常": b.get("anomaly"),
            "失败": b.get("failed"),
            "未处理": b.get("unprocessed"),
            "开始": b.get("created_at"),
            "结束": b.get("finished_at"),
        }
        for b in filtered
    ])
    st.dataframe(frame, width="stretch", hide_index=True)
    batch_labels = [f"{b['batch_id']} · v{b.get('product_version')} · {b.get('status')}"
                    for b in filtered]
    chosen_batch = st.selectbox("查看批次详情", batch_labels, key="product_batch_history_batch")
    try:
        payload = services.load_product_batch(
            filtered[batch_labels.index(chosen_batch)]["batch_id"])
    except Exception as exc:  # noqa: BLE001
        st.error(f"批次详情读取失败：{exc}")
        return
    _render_batch_detail(payload)


def _drift_items_frame(payload: dict) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "序号": item.get("index"),
            "文件": item.get("original_filename"),
            "状态": item.get("status"),
            "全局距离": item.get("global_score"),
            "空间均值": item.get("spatial_mean"),
            "空间p95": item.get("spatial_p95"),
            "亮度": item.get("brightness"),
            "对比度": item.get("contrast"),
            "错误": item.get("error"),
        }
        for item in payload.get("items") or []
    ])


def _render_drift_detail(payload: dict, *, scope: str = "history") -> None:
    level = payload.get("level")
    labels = {
        "normal": "正常/未观察到明显漂移",
        "observation": "观察/可能存在变化",
        "drift_warning": "漂移预警",
        "insufficient": "样本不足",
        "failed": "监测失败",
    }
    if level in ("drift_warning", "failed"):
        st.error(f"监测状态：{labels.get(level, level)}")
    elif level in ("observation", "insufficient"):
        st.warning(f"监测状态：{labels.get(level, level)}")
    else:
        st.success(f"监测状态：{labels.get(level, level)}")
    st.caption(
        f"产品：{payload.get('product_name')}（{payload.get('product_id')}）　"
        f"版本：v{payload.get('product_version')}　"
        f"manifest：{str(payload.get('manifest_sha256', ''))[:16]}…　"
        f"bank：{str(payload.get('bank_version', ''))[:32]}…"
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("窗口图片", payload.get("total", 0))
    c2.metric("成功", payload.get("success", 0))
    c3.metric("失败", payload.get("failed", 0))
    c4.metric("最小窗口", payload.get("min_window_samples", 5))
    if payload.get("reasons"):
        for reason in payload["reasons"]:
            st.write(f"- {reason}")
    st.caption(f"阈值来源：{payload.get('threshold_source')}　算法：{payload.get('algorithm_version')}")
    metrics = payload.get("metrics") or {}
    rows = []
    for key, label in (("global", "全局特征距离"), ("spatial_mean", "局部空间均值"),
                       ("spatial_p95", "局部空间p95"), ("brightness", "亮度"),
                       ("contrast", "对比度")):
        stats = metrics.get(key) or {}
        rows.append({
            "指标": label,
            "窗口均值": stats.get("mean"),
            "窗口标准差": stats.get("std"),
            "窗口p95": stats.get("p95"),
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.dataframe(_drift_items_frame(payload), width="stretch", hide_index=True)
    drift_items = [item for item in payload.get("items") or []
                   if item.get("status") == "success" and item.get("input_path")]
    if drift_items:
        labels = [f"#{item['index']} {item['original_filename']}" for item in drift_items]
        chosen = st.selectbox("选择漂移窗口中的图片",
                              labels, key=f"drift_feedback_item_{scope}_{payload['window_id']}")
        selected_item = drift_items[labels.index(chosen)]
        if st.button("提交该图人工审核",
                     key=f"drift_feedback_{scope}_{payload['window_id']}_{selected_item['index']}"):
            try:
                feedback = services.create_product_feedback_from_drift(
                    payload["product_id"], product_version=payload["product_version"],
                    window_id=payload["window_id"], item=selected_item,
                    drift_level=payload.get("level"),
                    drift_reasons=payload.get("reasons") or [],
                )
                st.success(f"已创建待审核反馈：{feedback['feedback_id']}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"创建反馈失败：{exc}")
    if level in ("observation", "drift_warning"):
        st.info(
            "建议检查光源、相机、拍摄角度、产品批次和生产环境；"
            "本页面不会自动更新记忆库、阈值、相机参数或 reference bank。"
        )
    window_id = payload["window_id"]
    try:
        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                "下载监测 JSON",
                data=services.drift_file_bytes(window_id, "window.json"),
                file_name=f"{window_id}.json", mime="application/json",
                key=f"drift_dl_json_{scope}_{window_id}",
            )
        with d2:
            st.download_button(
                "下载监测 CSV",
                data=services.drift_file_bytes(window_id, "results.csv"),
                file_name=f"{window_id}.csv", mime="text/csv",
                key=f"drift_dl_csv_{scope}_{window_id}",
            )
    except Exception as exc:  # noqa: BLE001
        st.warning(f"监测记录下载不可用：{exc}")


def render_drift_tab(root: Path) -> None:
    st.subheader("产品漂移监测")
    st.caption("窗口聚合监测；实验性阈值，不提供统计显著性/FPR 保证，不自动更新任何产品资产。")
    try:
        cards = services.list_product_cards(root)
    except Exception as exc:  # noqa: BLE001
        st.error(f"产品列表读取失败：{exc}")
        return
    detectable = [card for card in cards if card["detectable"]]
    if not detectable:
        st.warning("当前没有可监测产品。")
        return
    labels = [f"{card['name']} · {card['product_id'][:8]} · v{card['current_version']}"
              for card in detectable]
    selected = st.selectbox("选择监测产品", labels, key="drift_product")
    card = detectable[labels.index(selected)]
    st.caption(
        f"基线：v{card['current_version']}　manifest {str(card['manifest_sha256'])[:16]}…　"
        f"来源样本：{card.get('source_status', 'legacy')}"
    )
    if card.get("source_status") != "persisted":
        st.error("该产品来源样本未持久化（legacy），无法建立漂移基线。")
        return
    uploaded = st.file_uploader(
        "生产窗口图片（至少 5 张）", type=PRODUCT_TYPES,
        accept_multiple_files=True, key="drift_uploader",
    )
    uploads = _upload_items(uploaded)
    st.caption(f"已选择 {len(uploads)} 张；窗口目录：cache/product_drift")
    if st.button("执行漂移分析", type="primary", key="drift_analyze",
                 disabled=len(uploads) < 5):
        try:
            with st.spinner("正在提取窗口特征并与注册校准基线比较…"):
                payload = services.analyze_product_drift(
                    card["product_id"], uploads, root=root, device="auto",
                )
            st.session_state["drift_result"] = payload
            st.session_state.pop("drift_error", None)
        except Exception as exc:  # noqa: BLE001
            st.session_state["drift_error"] = f"{type(exc).__name__}: {exc}"
            st.session_state.pop("drift_result", None)
    error = st.session_state.get("drift_error")
    if error:
        st.error(f"漂移监测失败：{error}")
    payload = st.session_state.get("drift_result")
    if payload:
        _render_drift_detail(payload, scope="session")

    st.divider()
    st.subheader("漂移监测历史")
    try:
        windows = services.list_drift_windows(product_id=card["product_id"])
    except Exception as exc:  # noqa: BLE001
        st.error(f"监测历史读取失败：{exc}")
        return
    if not windows:
        st.info("暂无漂移监测记录。")
        return
    frame = pd.DataFrame([
        {
            "窗口": w["window_id"],
            "产品": w.get("product_name"),
            "版本": w.get("product_version"),
            "状态": w.get("level"),
            "总数": w.get("total"),
            "成功": w.get("success"),
            "失败": w.get("failed"),
            "时间": w.get("created_at"),
        }
        for w in windows
    ])
    st.dataframe(frame, width="stretch", hide_index=True)
    choices = [f"{w['window_id']} · v{w.get('product_version')} · {w.get('level')}"
               for w in windows]
    chosen = st.selectbox("查看监测窗口", choices, key="drift_history_window")
    try:
        historical = services.load_drift_window(
            windows[choices.index(chosen)]["window_id"])
    except Exception as exc:  # noqa: BLE001
        st.error(f"监测详情读取失败：{exc}")
        return
    _render_drift_detail(historical, scope="history")


def _feedback_frame(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "feedback_id": item["feedback_id"],
            "版本": item["product_version"],
            "来源": item["source_type"],
            "来源ID": item.get("source_id"),
            "原结论": item.get("image_status"),
            "分数": item.get("image_score"),
            "状态": item.get("status"),
            "审核结论": item.get("conclusion"),
            "时间": item.get("created_at"),
        }
        for item in records
    ])


def render_feedback_tab(root: Path) -> None:
    st.subheader("人工审核与受控更新")
    st.caption("只有 confirmed 的正常/异常反馈可作为更新候选；环境变化/拒绝不得进入记忆。")
    try:
        cards = services.list_product_cards(root)
    except Exception as exc:  # noqa: BLE001
        st.error(f"产品列表读取失败：{exc}")
        return
    if not cards:
        st.info("暂无产品。")
        return
    labels = [f"{c['name']} · {c['product_id'][:8]} · v{c['current_version']}" for c in cards]
    selected = st.selectbox("审核产品", labels, key="feedback_product")
    card = cards[labels.index(selected)]
    pid = card["product_id"]
    try:
        records = services.list_product_feedback(pid, root=root)
    except Exception as exc:  # noqa: BLE001
        st.error(f"反馈读取失败：{exc}")
        return

    st.subheader("待审核队列")
    if not records:
        st.info("暂无反馈。可在单图检测结果下提交人工审核，或通过批次/漂移结果创建。")
    else:
        st.dataframe(_feedback_frame(records), width="stretch", hide_index=True)
        pending = [item for item in records if item["status"] == "pending"]
        if pending:
            ids = [item["feedback_id"] for item in pending]
            chosen_id = st.selectbox("选择待审核反馈", ids, key="feedback_pending")
            item = next(x for x in pending if x["feedback_id"] == chosen_id)
            c1, c2 = st.columns([1, 1])
            with c1:
                try:
                    st.image(services.feedback_image_bytes(pid, item, root=root),
                             caption=f"{item.get('original_filename')} · {item.get('source_type')}",
                             width="stretch")
                except Exception as exc:  # noqa: BLE001
                    st.warning(f"图片不可用：{exc}")
            with c2:
                st.caption(f"版本 v{item.get('product_version')}　SHA {item.get('image_sha256')}")
                st.caption(f"原检测：{item.get('image_status')}　分数 {item.get('image_score')}　阈值 {item.get('image_threshold')}")
                if item.get("drift_level"):
                    st.caption(f"漂移状态：{item.get('drift_level')}　原因：{item.get('drift_reasons')}")
            conclusion = st.radio(
                "审核结论", options=[
                    ("normal", "确认正常"), ("anomaly", "确认异常"),
                    ("environment", "环境/成像条件变化"), ("reject", "拒绝/无法判断"),
                ], format_func=lambda x: x[1], key=f"feedback_conclusion_{chosen_id}",
            )[0]
            note = st.text_area("审核说明（可选）", key=f"feedback_note_{chosen_id}")
            if st.button("提交审核", key=f"feedback_submit_{chosen_id}"):
                try:
                    services.review_product_feedback(pid, chosen_id, conclusion, note, root=root)
                    st.success("审核已保存")
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"审核失败：{exc}")

    st.divider()
    st.subheader("更新候选与独立验证")
    confirmed = [item for item in records if item["status"] == "confirmed"]
    if not confirmed:
        st.info("暂无 confirmed 反馈。")
    else:
        label_map = {
            item["feedback_id"]:
            f"{item['feedback_id'][:8]} · {item.get('source_type')} · {item.get('conclusion')} · v{item['product_version']}"
            for item in confirmed
        }
        chosen_ids = st.multiselect("选择已确认反馈", options=list(label_map),
                                    format_func=lambda x: label_map[x],
                                    key="feedback_update_ids")
        normal_uploads = st.file_uploader(
            "独立验证正常图（至少 5 张，必须与参考/校准/更新图去重）",
            type=PRODUCT_TYPES, accept_multiple_files=True, key="validation_normal",
        )
        anomaly_uploads = st.file_uploader(
            "独立验证异常图（至少 5 张，必须与参考/校准/更新图去重）",
            type=PRODUCT_TYPES, accept_multiple_files=True, key="validation_anomaly",
        )
        normals = _upload_items(normal_uploads)
        anomalies = _upload_items(anomaly_uploads)
        note = st.text_area("更新说明（可选）", key="feedback_update_note")
        disabled = (not chosen_ids or len(normals) < 5 or len(anomalies) < 5)
        if st.button("创建并验证新版本", type="primary", disabled=disabled,
                     key="feedback_prepare_update"):
            try:
                with st.spinner("正在创建新版本并在独立验证集上比较…"):
                    report = services.prepare_product_update(
                        pid, chosen_ids, normals, anomalies, note=note, root=root,
                    )
                st.session_state["p5_update_report"] = report
                st.session_state.pop("p5_update_error", None)
            except Exception as exc:  # noqa: BLE001
                st.session_state["p5_update_error"] = f"{type(exc).__name__}: {exc}"
                st.session_state.pop("p5_update_report", None)
        error = st.session_state.get("p5_update_error")
        if error:
            st.error(f"更新失败：{error}")
        report = st.session_state.get("p5_update_report")
        if report:
            st.write(f"更新 ID：{report['update_id']}　旧 v{report['old_version']} → 新 v{report['new_version']}")
            st.write(f"验证结论：**{report['verdict']}**　{report['reasons']}")
            metric_rows = []
            for key, label in (("false_positives", "正常误报"),
                               ("false_negatives", "异常漏报"),
                               ("auroc", "AUROC"), ("average_precision", "AP"),
                               ("f1", "F1"), ("threshold", "阈值")):
                metric_rows.append({"指标": label,
                                    "旧版本": report["old_metrics"][key],
                                    "新版本": report["new_metrics"][key]})
            st.dataframe(pd.DataFrame(metric_rows), width="stretch", hide_index=True)
            if report["verdict"] == "pass" and not report.get("activated"):
                if st.button("确认激活新版本", type="primary", key="p5_activate"):
                    try:
                        services.activate_product_update(pid, report["update_id"], root=root)
                        st.success("新版本已显式激活")
                        st.session_state.pop("p5_update_report", None)
                        st.rerun()
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"激活失败：{exc}")
            elif report.get("activated"):
                st.success("该更新已激活")
            else:
                st.warning("验证未通过/不充分，保留旧 current_version。")

    st.divider()
    st.subheader("更新历史")
    try:
        updates = services.list_product_updates(pid, root=root)
    except Exception as exc:  # noqa: BLE001
        st.error(f"更新历史读取失败：{exc}")
        return
    if updates:
        st.dataframe(pd.DataFrame([
            {"更新": u["update_id"], "旧版本": u["old_version"], "新版本": u["new_version"],
             "验证": u["verdict"], "已激活": u["activated"], "时间": u["created_at"]}
            for u in updates
        ]), width="stretch", hide_index=True)
        update_labels = {
            u["update_id"]: f"{u['update_id'][:8]} · v{u['old_version']}→v{u['new_version']} · {u['verdict']}"
            for u in updates
        }
        chosen_update = st.selectbox("查看/激活更新", list(update_labels),
                                     format_func=lambda x: update_labels[x],
                                     key="p5_update_history")
        chosen_report = next(u for u in updates if u["update_id"] == chosen_update)
        if chosen_report["verdict"] == "pass" and not chosen_report["activated"]:
            if st.button("激活该更新", type="primary", key="p5_activate_history"):
                try:
                    services.activate_product_update(pid, chosen_update, root=root)
                    st.success("新版本已激活")
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"激活失败：{exc}")
    else:
        st.info("暂无更新记录。")


def render_detection_tab(root: Path) -> None:
    st.subheader("产品单图检测")
    st.caption(f"产品根目录：{root}　检测只读产品数据，使用 current_version 权威版本。")
    try:
        cards = services.list_product_cards(root)
    except Exception as exc:  # noqa: BLE001
        st.error(f"产品列表读取失败：{exc}")
        return

    if not cards:
        st.info("产品目录为空。请先在「产品注册」页注册产品。")
        return
    _product_table(cards)

    detectable = [card for card in cards if card["detectable"]]
    if not detectable:
        st.warning("当前没有可检测产品；请查看上表的不可检测原因。")
        return
    labels = [
        f"{card['name']} · {card['product_id'][:8]} · v{card['current_version']}"
        for card in detectable
    ]
    selected_label = st.selectbox("选择产品", labels, key="product_select")
    card = detectable[labels.index(selected_label)]

    c1, c2, c3 = st.columns(3)
    c1.metric("当前版本", f"v{card['current_version']}")
    c2.metric("image_threshold", f"{card['image_threshold']:.6f}")
    c3.metric("manifest", card["manifest_sha256"][:12] + "…")
    st.caption(f"product_id：{card['product_id']}　注册时间：{card['created_at']}")
    st.caption("完整性校验：通过；检测引擎会在加载时再次校验并持有产品锁。")
    if card.get("source_status") == "persisted":
        with st.expander(f"来源样本（{len(card.get('source_samples') or [])} 张，已持久化）",
                         expanded=False):
            source_frame = pd.DataFrame([
                {
                    "role": item.get("role"),
                    "相对路径": item.get("rel_path"),
                    "原始文件名": item.get("original_filename"),
                    "SHA256": item.get("sha256"),
                    "大小(B)": item.get("size_bytes"),
                }
                for item in card.get("source_samples") or []
            ])
            st.dataframe(source_frame, width="stretch", hide_index=True)
    else:
        st.caption("来源样本：legacy/不可用（旧产品未持久化原始图片）。")

    device = st.selectbox("推理设备", ["auto", "cuda", "cpu"], index=0, key="product_device")
    uploaded = st.file_uploader(
        "待检测图片（PNG / JPG / JPEG / BMP）", type=PRODUCT_TYPES,
        key="product_detect_uploader",
    )
    image_error = None
    image_sha = None
    desired_signature = None
    if uploaded is not None:
        try:
            data = uploaded.getvalue()
            services.validate_image_bytes(data, uploaded.name)
            image_sha = hashlib.sha256(data).hexdigest()
            desired_signature = services.product_signature(
                card["product_id"], card["current_version"], image_sha, root, device
            )
            st.image(uploaded, caption=f"待检测：{uploaded.name}", width="stretch")
        except services.ImageValidationError as exc:
            image_error = str(exc)
            st.error(f"图片不可用：{exc}")

    clicked = st.button(
        "开始产品检测", type="primary", key="product_detect_button",
        disabled=uploaded is None or image_error is not None,
    )
    if clicked and uploaded is not None:
        data = uploaded.getvalue()
        cached = st.session_state.get("product_detection")
        if cached is not None and desired_signature is not None and cached.get("signature") == desired_signature:
            st.info("相同产品版本、图片与设备的结果已存在，直接复用，未重复执行检测。")
        else:
            try:
                services.validate_image_bytes(data, uploaded.name)
                with st.spinner("正在加载产品记忆库、提取特征并计算异常分数…"):
                    state = services.run_product_detection(
                        card["product_id"], data, uploaded.name, root=root, device=device
                    )
                st.session_state["product_detection"] = state
                st.session_state.pop("product_detection_error", None)
            except (services.ImageValidationError, ProductDetectorError) as exc:
                st.session_state["product_detection_error"] = f"{type(exc).__name__}: {exc}"
                st.session_state.pop("product_detection", None)
            except Exception as exc:  # noqa: BLE001
                st.session_state["product_detection_error"] = f"{type(exc).__name__}: {exc}"
                st.session_state.pop("product_detection", None)

    st.divider()
    error = st.session_state.get("product_detection_error")
    if error:
        st.error(f"检测失败：{error}")
    state = st.session_state.get("product_detection")
    if state is None:
        st.info("上传一张图片并点击「开始产品检测」。")
        return
    if state["result"].product_id != card["product_id"]:
        st.info("已更换选择的产品，请点击「开始产品检测」获取当前产品的检测结果。")
        return
    if uploaded is None:
        st.info("当前没有待检测图片；上次结果仍保留，可重新上传后再次检测。")
    elif image_sha is not None and state["result"].input_image_sha256 != image_sha:
        st.info("已更换待检测图片，请点击「开始产品检测」获取新结果。")
        return
    if state["result"].version != card["current_version"]:
        st.warning(
            f"当前产品指针已变为 v{card['current_version']}，下面显示的是本次实际使用的 "
            f"v{state['result'].version} 结果；如需最新版本，请重新检测。"
        )
    _show_detection_result(state)
