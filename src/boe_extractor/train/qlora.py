"""QLoRA con Unsloth sobre el silver (en Colab, GPU T4).

Uso: python -m boe_extractor.train.qlora --datos sft/ --salida qlora-v2/

Guarda checkpoints cada 50 pasos en <salida>/checkpoints y, si la sesión de Colab se corta,
al relanzar sigue desde el último. Al terminar deja el adaptador en <salida>/adaptador y la
curva de pérdida en <salida>/historial.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from boe_extractor.infer import MAX_LEN, MODELO

HIPER = {
    "r": 16, "lora_alpha": 16, "lr": 2e-4, "epochs": 2, "batch": 1, "grad_accum": 8,
}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", type=Path, required=True)
    ap.add_argument("--salida", type=Path, required=True)
    ap.add_argument("--epochs", type=float, default=HIPER["epochs"])
    args = ap.parse_args()

    from unsloth import FastLanguageModel  # noqa: I001  (antes que transformers/trl)
    from unsloth.chat_templates import train_on_responses_only

    import torch
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    model, tok = FastLanguageModel.from_pretrained(
        MODELO, max_seq_length=MAX_LEN, load_in_4bit=True
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=HIPER["r"],
        lora_alpha=HIPER["lora_alpha"],
        lora_dropout=0,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    # Sin evaluación durante el entrenamiento: con 8k tokens los logits en fp32 no caben en la
    # T4 (OOM). La medida que cuenta es boe-eval sobre el gold.
    ds = load_dataset("json", data_files=str(args.datos / "train.jsonl"), split="train").map(
        lambda r: {"text": tok.apply_chat_template(r["messages"], tokenize=False)}
    )
    ckpt = args.salida / "checkpoints"
    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=ds,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=MAX_LEN,
            per_device_train_batch_size=HIPER["batch"],
            gradient_accumulation_steps=HIPER["grad_accum"],
            num_train_epochs=args.epochs,
            learning_rate=HIPER["lr"],
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            optim="adamw_8bit",
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            save_steps=50,
            save_total_limit=2,
            output_dir=str(ckpt),
            report_to="none",
            seed=3407,
        ),
    )
    # Solo se aprende la respuesta (el JSON), no a repetir el documento
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )
    trainer.train(resume_from_checkpoint=ckpt.exists() and any(ckpt.glob("checkpoint-*")))
    model.save_pretrained(args.salida / "adaptador")
    tok.save_pretrained(args.salida / "adaptador")
    historial = {"hiper": HIPER | {"epochs": args.epochs}, "log": trainer.state.log_history}
    (args.salida / "historial.json").write_text(json.dumps(historial, indent=1))
    print(f"Adaptador → {args.salida / 'adaptador'}")


if __name__ == "__main__":
    main()
