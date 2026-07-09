import sys
from huggingface_hub import login, whoami

# Try the second part (looks like two tokens got concatenated)
candidates = [
    "hf_OtwFwqjSYsLLvzsBUsPIqxPeVBTvXdNMyzy",
    "hf_OwwjYsLLzsBUPqPeBTXdMzhf_OtwFwqjSYsLLvzsBUsPIqxPeVBTvXdNMyzy",
]

for tok in candidates:
    try:
        login(token=tok, add_to_git_credential=False)
        info = whoami()
        print(f"Logged in as: {info['name']}")
        sys.exit(0)
    except Exception as e:
        print(f"Failed ({tok[:12]}...): {e}")

print("All attempts failed. Please check your token.")
sys.exit(1)
