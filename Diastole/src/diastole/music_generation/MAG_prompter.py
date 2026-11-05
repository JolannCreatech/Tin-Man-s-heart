import os
def initialize_music_generation():
    os.system('uv run mrt mlx init')

def generate_music(prompt, duration=10):
    command = (
        f'uv run mrt mlx generate '
        f'--prompt "{prompt}" '
        f'--duration {duration} '
        f'--model mrt2_base'
    )

    os.system(command)


generate_music(
    "If I only had a heart",
    duration=10
)