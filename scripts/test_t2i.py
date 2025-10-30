from word2png_function import text_to_images
CONFIG_EN_PATH = 'Glyph/config/config_en.json'
OUTPUT_DIR = 'output_images'
INPUT_FILE = '/home/aiscuser/AgentOCR/data/hotpotqa/eval_50.json'

# Read text from file
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    text = f.read()[:1000]

# Convert text to images
images = text_to_images(
    text=text,
    output_dir=OUTPUT_DIR,
    config_path=CONFIG_EN_PATH,
    unique_id='test_001'
)

print(f"\nGenerated {len(images)} image(s):")
for img_path in images:
    print(f"  {img_path}")