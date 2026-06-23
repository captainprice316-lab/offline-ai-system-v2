import json, pathlib

root = pathlib.Path('finetune_runs')
langs = ['hi', 'ne', 'zh', 'ur', 'pa', 'ps']

all_data = {}

for lang in langs:
    adapter = root / lang / 'adapter'
    if not adapter.exists():
        print(f'=== {lang}: missing ===')
        continue
    checkpoints = sorted(adapter.glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[1]))
    print(f'=== {lang} checkpoints: {[c.name for c in checkpoints]} ===')
    lang_history = []
    for ckpt in checkpoints:
        ts = ckpt / 'trainer_state.json'
        if not ts.exists():
            continue
        data = json.loads(ts.read_text(encoding='utf-8'))
        for e in data.get('log_history', []):
            if 'eval_wer' in e:
                step = e['step']
                wer = e['eval_wer']
                eloss = e.get('eval_loss', None)
                print(f'  {ckpt.name}  step={step}  wer={wer:.4f}  eval_loss={eloss}')
                lang_history.append({'step': step, 'wer': wer, 'eval_loss': eloss})
            elif 'loss' in e:
                step = e['step']
                tloss = e['loss']
                gn = e.get('grad_norm', None)
                print(f'  step={step:4d}  train_loss={tloss:.4f}  grad_norm={gn}')
    all_data[lang] = lang_history

print('\n\n=== SUMMARY ===')
for lang, history in all_data.items():
    if history:
        best = min(history, key=lambda x: x['wer'])
        print(f'{lang}: best WER = {best["wer"]:.4f} at step {best["step"]}')
    else:
        print(f'{lang}: no eval data found')
