import gradio as gr
from bpe_tokenizer import SanskritBPETokenizer
import json
import random
import colorsys

# Load the trained tokenizer
with open('tokenizer_config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

tokenizer = SanskritBPETokenizer(vocab_size=4999)
tokenizer.vocab = config['vocab']
tokenizer.merges = {tuple(k.split('|')): v for k, v in config['merges'].items()}
tokenizer.inverse_vocab = config['inverse_vocab']

# Load example texts
with open('examples/example1.txt', 'r', encoding='utf-8') as f:
    example1 = f.read().strip()
    
with open('examples/example2.txt', 'r', encoding='utf-8') as f:
    example2 = f.read().strip()

def generate_distinct_colors(n):
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.7 + random.random() * 0.3
        value = 0.8 + random.random() * 0.2
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        hex_color = "#{:02x}{:02x}{:02x}".format(
            int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
        )
        colors.append(hex_color)
    return colors

def process_text(text):
    try:
        # Get tokens and their positions
        tokens, positions = tokenizer.encode_with_positions(text)
        
        # Generate colors for visualization
        unique_tokens = list(set(tokens))
        colors = generate_distinct_colors(len(unique_tokens))
        token_colors = dict(zip(unique_tokens, colors))
        
        # Create HTML visualization
        html_parts = []
        current_pos = 0
        
        for token, (start, end) in zip(tokens, positions):
            if start > current_pos:
                html_parts.append(text[current_pos:start])
            
            token_text = text[start:end]
            color = token_colors[token]
            html_parts.append(f'<span style="background-color: {color}; padding: 0 2px; border-radius: 3px;" title="Token ID: {token}">{token_text}</span>')
            
            current_pos = end
        
        if current_pos < len(text):
            html_parts.append(text[current_pos:])
        
        html_output = ''.join(html_parts)
        
        # Simplified token information
        token_info = f"Token count: {len(tokens)}\nTokens: {tokens}"
        
        return (
            html_output,
            token_info
        )
    except Exception as e:
        return (
            f"<span style='color: red'>Error processing text: {str(e)}</span>",
            f"Error: {str(e)}"
        )

# Create Gradio interface
iface = gr.Interface(
    fn=process_text,
    inputs=gr.Textbox(label="Enter Sanskrit Text", lines=3),
    outputs=[
        gr.HTML(label="Visualization"),
        gr.Textbox(label="Token Information", lines=10)
    ],
    title="Sanskrit BPE Tokenizer Visualizer",
    description="Enter Sanskrit text to see how it gets tokenized, with color-coded visualization",
    examples=[
        [example1],
        [example2]
    ],
    flagging_options=None
)

iface.launch(share=False)