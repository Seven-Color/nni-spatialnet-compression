with open(r'C:\Users\wuxiukun\.openclaw\workspace-codex\nni_spatialnet_compression\step6_complete_demo.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find and replace the pattern - use _output_ instead of output
old = """'quant_dtype': f'int{quant_bits}',
                'target_names': ['weight', 'output'],
                'target_settings': {
                    'weight': {'quant_dtype': f'int{quant_bits}'},
                    'output': {'quant_dtype': f'int{quant_bits}'}
                }"""

new = """'quant_dtype': f'int{quant_bits}',
                'target_names': ['weight', '_output_'],
                'target_settings': {
                    'weight': {'quant_dtype': f'int{quant_bits}'},
                    '_output_': {'quant_dtype': f'int{quant_bits}'}
                }"""

content = content.replace(old, new)

with open(r'C:\Users\wuxiukun\.openclaw\workspace-codex\nni_spatialnet_compression\step6_complete_demo.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Done')
