#!/usr/bin/env python3
# this_file: test_minimal_custom_llm.py
"""Minimal test to verify LiteLLM CustomLLM registration works."""

import litellm
from litellm import CustomLLM, completion

# Enable debug to see what's happening
litellm._turn_on_debug()


class MyCustomLLM(CustomLLM):
    def completion(self, *args, **kwargs) -> litellm.ModelResponse:
        print(
            f"🎯 MyCustomLLM.completion called with args={len(args)}, kwargs keys={list(kwargs.keys())}"
        )
        return litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello world"}],
            mock_response="Hi!",
        )


my_custom_llm = MyCustomLLM()

print("📝 Setting custom provider map...")
litellm.custom_provider_map = [
    {"provider": "my-custom-llm", "custom_handler": my_custom_llm}
]

print(f"✅ Registered: {litellm.custom_provider_map}")

print("🚀 Testing completion call...")
try:
    resp = completion(
        model="my-custom-llm/my-fake-model",
        messages=[{"role": "user", "content": "Hello world!"}],
    )
    print(f"✅ Success! Response: {resp.choices[0].message.content}")
except Exception as e:
    print(f"❌ Error: {e}")
    print(f"   Type: {type(e)}")
