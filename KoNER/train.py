import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from KoNER import (
    initialize_directories,
    get_ko_ner_configuration,
    fix_torch_seed,

    NerTokenizer,
    NerDataModule,
    NerModelTrainer
)


def train(config_path, config_file):
    fix_torch_seed()

    config = get_ko_ner_configuration(
        config_path=config_path,
        config_file=config_file)

    initialize_directories(config=config)

    model_type = f"{config.model.embedding_layer}-{config.model.middle_layer}-{config.model.output_layer}"
    wandb_name = f"{model_type} batch:{config.parameters.batch_size} lr:{config.parameters.learning_rate} weight_decay: {config.parameters.weight_decay}"

    wandb_logger = WandbLogger(
        name=wandb_name,
        project="KoNER"
    )

    ner_tokenizer = NerTokenizer(tokenizer_name=config.tokenizer.name)
    ner_data_module = NerDataModule(
        config=config,
        model_type=model_type,
        tokenizer=ner_tokenizer)

    ner_model_trainer = NerModelTrainer(
        config=config,

        l2i=ner_data_module.get_l2i(),
        i2l=ner_data_module.get_i2l(),

        token_pad_id=ner_data_module.get_token_pad_id(),
        label_pad_id=ner_data_module.get_label_pad_id(),

        label_start_id=ner_data_module.get_label_start_id(),
        label_end_id=ner_data_module.get_label_end_id())

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_f1_score",
        dirpath=config.path.model,
        filename=f"ner-{model_type}" + "-{val_f1_score:0.4f}-{val_loss:0.4f}",
        save_top_k=1,
        mode="max")

    # Train
    trainer = pl.Trainer(
        max_epochs=config.parameters.epoch,
        gpus=2,
        accelerator="dp",
        precision=16,
        logger=wandb_logger,
        callbacks=[checkpoint_callback])

    trainer.fit(ner_model_trainer, ner_data_module)

if __name__ == "__main__":
    train(
        config_path="../config/ner",
        config_file="ner.cfg"
    )