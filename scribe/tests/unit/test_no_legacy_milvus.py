"""Phase A8 cleanup guards: legacy langchain-milvus stack must be gone.

The Milvus integration lives exclusively in scribe.backends.milvus_backend
(direct pymilvus MilvusClient). The old scribe.milvus package and the
langchain-milvus dependency must not come back.
"""

import importlib.util


class TestLegacyMilvusStackRemoved:
    def test_scribe_milvus_package_is_deleted(self):
        assert importlib.util.find_spec("scribe.milvus") is None, (
            "scribe/milvus/ (connector/content_manager/retriever/schema) is legacy "
            "langchain-milvus code; use scribe.backends.milvus_backend instead"
        )

    def test_langchain_milvus_dependency_is_removed(self):
        assert importlib.util.find_spec("langchain_milvus") is None, (
            "langchain-milvus must not be installed; Milvus access goes through "
            "pymilvus MilvusClient (scribe.backends.milvus_backend)"
        )

    def test_facade_has_no_test_mode_plumbing(self):
        from scribe.scribe_milvus import SCRIBE

        assert not hasattr(SCRIBE, "_test_mode"), (
            "_test_mode plumbing was removed in Phase A8; tests mock the facade "
            "via scribe.tests.mocks.mock_scribe_service instead"
        )
