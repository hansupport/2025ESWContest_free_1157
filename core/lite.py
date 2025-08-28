# core/lite.py  (Python 3.6 호환, 지연 임포트 래퍼)
from __future__ import print_function
import importlib

def _m(name):
    return importlib.import_module(name)

def preload_all():
    """선로딩해서 첫 호출 지연을 0에 가깝게."""
    _m("core.dm"); _m("core.emb"); _m("core.models"); _m("core.smooth"); _m("core.storage")

class DM(object):
    @staticmethod
    def lock():
        return _m("core.dm").DM_LOCK

    @staticmethod
    def open_persistent(camera, prefer_res, prefer_fps):
        return _m("core.dm").open_persistent(camera, prefer_res, prefer_fps)

    @staticmethod
    def scan_fast4(handle, rois, timeout_s, debug=False, trace_id=None):
        return _m("core.dm").scan_fast4(handle, rois, timeout_s, debug, trace_id)

    @staticmethod
    def close_persistent():
        return _m("core.dm").close_persistent()

class Emb(object):
    @staticmethod
    def build_embedder(S):
        return _m("core.emb").build_embedder(S)

    @staticmethod
    def warmup_shared(emb, S, dm_handle, dm_lock, frames, pregrab):
        return _m("core.emb").warmup_shared(emb, S, dm_handle, dm_lock, frames, pregrab)

    @staticmethod
    def embed_one_frame_shared(emb, S, dm_handle, dm_lock, pregrab=3):
        return _m("core.emb").embed_one_frame_shared(emb, S, dm_handle, dm_lock, pregrab)

class Models(object):
    @staticmethod
    def InferenceEngine(S):
        return _m("core.models").InferenceEngine(S)

    @staticmethod
    def ProbSmoother(window, min_votes):
        return _m("core.smooth").ProbSmoother(window=window, min_votes=min_votes)

class Storage(object):
    @staticmethod
    def open_db(db_path):
        return _m("core.storage").open_db(db_path)

    @staticmethod
    def on_sample_record(conn, feat15, emb128, product_id, has_label, origin):
        return _m("core.storage").on_sample_record(conn, feat15, emb128, product_id, has_label, origin)
