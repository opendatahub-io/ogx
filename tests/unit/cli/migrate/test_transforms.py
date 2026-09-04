# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for the pure Praxis transforms (no DB).

Fixtures are built from the real ogx_api models so the stored-blob shape matches
what OGX's responses/conversations stores actually persist.
"""

import json

from ogx.cli.migrate.praxis.target import (
    TenantDeriver,
    transform_conversation,
    transform_item,
    transform_legacy_inline_item,
    transform_message_only_conversation,
    transform_response,
)
from ogx_api import (
    OpenAIResponseInputMessageContentText,
    OpenAIResponseMessage,
    OpenAIResponseObject,
    OpenAIResponseOutputMessageContentOutputText,
)


def _input_message(text: str = "hi", item_id: str = "msg_in_1") -> dict:
    return OpenAIResponseMessage(
        id=item_id,
        role="user",
        content=[OpenAIResponseInputMessageContentText(text=text)],
    ).model_dump()


def _stored_response_blob(
    *,
    response_id: str = "resp_1",
    created_at: int = 111,
    model: str = "gpt-4o",
    input_items: list[dict] | None = None,
    messages: list[dict] | None = None,
    input_storage_mode: str | None = None,
) -> dict:
    """Emulate the blob OGX stores in ``openai_responses.response_object``:
    ``OpenAIResponseObject.model_dump()`` plus injected input/messages/mode."""
    out = OpenAIResponseMessage(
        id="msg_out_1",
        role="assistant",
        status="completed",
        content=[OpenAIResponseOutputMessageContentOutputText(text="hello")],
    )
    obj = OpenAIResponseObject(
        id=response_id,
        created_at=created_at,
        model=model,
        object="response",
        output=[out],
        status="completed",
        store=True,
    )
    blob = obj.model_dump()
    blob["input"] = input_items if input_items is not None else [_input_message()]
    if messages is not None:
        blob["messages"] = messages
    if input_storage_mode is not None:
        blob["input_storage_mode"] = input_storage_mode
    return blob


class TestTransformResponse:
    def test_response_object_strips_internal_fields_and_round_trips(self):
        blob = _stored_response_blob(messages=[{"role": "user", "content": "hi"}])
        row = {"id": "resp_1", "created_at": 111, "model": "gpt-4o", "response_object": blob, "owner_principal": "o1"}

        result = transform_response(row, TenantDeriver())

        public = json.loads(result.response_object)
        assert "input" not in public
        assert "messages" not in public
        assert "input_storage_mode" not in public
        # response_object is an OpenAI-compliant public object.
        OpenAIResponseObject(**public)

    def test_input_is_raw_blob_input_not_reconstructed(self):
        # An incremental response carries only this turn's input in the blob.
        incremental_input = [_input_message(text="only this turn", item_id="msg_incr")]
        blob = _stored_response_blob(input_items=incremental_input, input_storage_mode="incremental")
        row = {
            "id": "resp_1",
            "created_at": 111,
            "model": "gpt-4o",
            "response_object": blob,
            "owner_principal": "o1",
        }

        result = transform_response(row, TenantDeriver())

        # input is copied verbatim from the blob — NOT expanded via ancestry walking.
        assert json.loads(result.input) == incremental_input

    def test_missing_messages_defaults_to_empty_list(self):
        blob = _stored_response_blob(messages=None)  # no messages key at all
        assert "messages" not in blob
        row = {"id": "resp_1", "created_at": 111, "model": "gpt-4o", "response_object": blob, "owner_principal": "o1"}

        result = transform_response(row, TenantDeriver())

        assert result.messages == "[]"

    def test_messages_copied_verbatim(self):
        messages = [{"role": "system", "content": "be nice"}, {"role": "user", "content": "hi"}]
        blob = _stored_response_blob(messages=messages)
        row = {"id": "resp_1", "created_at": 111, "model": "gpt-4o", "response_object": blob, "owner_principal": "o1"}

        result = transform_response(row, TenantDeriver())

        assert json.loads(result.messages) == messages

    def test_scalar_fields_and_tenant(self):
        blob = _stored_response_blob(response_id="resp_9", created_at=999, model="gpt-4o-mini")
        row = {
            "id": "resp_9",
            "created_at": 999,
            "model": "gpt-4o-mini",
            "response_object": blob,
            "owner_principal": "tenant-x",
            "tenant_id": "tenant-x",
        }

        result = transform_response(row, TenantDeriver())

        assert result.id == "resp_9"
        assert result.created_at == 999
        assert result.model == "gpt-4o-mini"
        assert result.tenant_id == "tenant-x"

    def test_as_row_positional_order(self):
        blob = _stored_response_blob()
        row = {"id": "resp_1", "created_at": 111, "model": "gpt-4o", "response_object": blob, "owner_principal": "o1"}
        result = transform_response(row, TenantDeriver())
        # (id, tenant_id, created_at, model, response_object, input, messages)
        as_row = result.as_row()
        assert as_row[0] == "resp_1"
        assert as_row[1] == "o1"
        assert as_row[2] == 111
        assert as_row[3] == "gpt-4o"
        assert as_row[5] == result.input
        assert as_row[6] == result.messages


class TestTransformConversation:
    def test_conversation_joined_with_messages(self):
        conv_row = {"id": "conv_1", "created_at": 100, "metadata": {"k": "v"}, "owner_principal": "o1"}
        messages_row = {
            "conversation_id": "conv_1",
            "messages": [{"role": "user", "content": "hi"}],
            "owner_principal": "o1",
        }

        result = transform_conversation(conv_row, messages_row, TenantDeriver())

        assert result.conversation_id == "conv_1"
        assert result.created_at == 100
        assert result.tenant_id == "o1"
        assert json.loads(result.metadata) == {"k": "v"}
        assert json.loads(result.messages) == [{"role": "user", "content": "hi"}]

    def test_conversation_without_messages_defaults(self):
        conv_row = {"id": "conv_1", "created_at": 100, "metadata": None, "owner_principal": "o1"}

        result = transform_conversation(conv_row, None, TenantDeriver())

        assert result.messages == "[]"
        assert result.metadata == "{}"

    def test_mismatched_tenants_between_joined_rows_raises(self):
        conv_row = {"id": "conv_1", "created_at": 100, "metadata": {"k": "v"}, "owner_principal": "o1"}
        messages_row = {
            "conversation_id": "conv_1",
            "messages": [{"role": "user", "content": "hi"}],
            "owner_principal": "o2",
        }

        import pytest

        with pytest.raises(ValueError, match="disagree|derives"):
            transform_conversation(conv_row, messages_row, TenantDeriver())

    def test_message_only_orphan_synthesizes_row(self):
        messages_row = {
            "conversation_id": "conv_orphan",
            "messages": [{"role": "assistant", "content": "hey"}],
            "owner_principal": "o2",
        }

        result = transform_message_only_conversation(messages_row, TenantDeriver(), orphan_created_at=555)

        assert result.conversation_id == "conv_orphan"
        assert result.created_at == 555
        assert result.tenant_id == "o2"
        assert result.metadata == "{}"
        assert json.loads(result.messages) == [{"role": "assistant", "content": "hey"}]


class TestTransformItem:
    def test_item_straight_copy(self):
        row = {
            "id": "item_1",
            "conversation_id": "conv_1",
            "created_at": 200,
            "sort_order": 3,
            "item_data": {"type": "message", "id": "item_1"},
            "owner_principal": "o1",
        }

        result = transform_item(row, TenantDeriver())

        assert result.item_id == "item_1"
        assert result.conversation_id == "conv_1"
        assert result.created_at == 200
        assert result.position == 3
        assert json.loads(result.item_data) == {"type": "message", "id": "item_1"}

    def test_legacy_null_sort_order_becomes_zero(self):
        row = {
            "id": "item_2",
            "conversation_id": "conv_1",
            "created_at": 200,
            "sort_order": None,
            "item_data": {"type": "message", "id": "item_2"},
            "owner_principal": "o1",
        }

        result = transform_item(row, TenantDeriver())

        assert result.position == 0

    def test_item_as_row_positional_order(self):
        row = {
            "id": "item_1",
            "conversation_id": "conv_1",
            "created_at": 200,
            "sort_order": 3,
            "item_data": {"type": "message"},
            "owner_principal": "o1",
        }
        as_row = transform_item(row, TenantDeriver()).as_row()
        # (item_id, tenant_id, conversation_id, item_data, created_at, position)
        assert as_row[0] == "item_1"
        assert as_row[1] == "o1"
        assert as_row[2] == "conv_1"
        assert as_row[4] == 200
        assert as_row[5] == 3


class TestTransformLegacyInlineItem:
    def test_backfill_inline_item(self):
        conv_row = {"id": "conv_1", "created_at": 100, "owner_principal": "o1"}
        element = {"id": "item_legacy", "type": "message", "role": "user"}

        result = transform_legacy_inline_item(element, position=2, conv_row=conv_row, tenant=TenantDeriver())

        assert result.item_id == "item_legacy"
        assert result.conversation_id == "conv_1"
        assert result.created_at == 100
        assert result.position == 2
        assert result.tenant_id == "o1"
        assert json.loads(result.item_data) == element

    def test_backfill_requires_element_id(self):
        conv_row = {"id": "conv_1", "created_at": 100, "owner_principal": "o1"}
        import pytest

        with pytest.raises(ValueError, match="no 'id'"):
            transform_legacy_inline_item({"type": "message"}, position=0, conv_row=conv_row, tenant=TenantDeriver())
