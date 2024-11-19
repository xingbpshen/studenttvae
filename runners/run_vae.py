import comet_ml
import torch
from comet_ml.integration.pytorch import log_model
from datasets import get_dataset
from models.vae import VAE
import torch.utils.data
from functions import get_optimizer
from functions.losses import calculate_loss
from functions.importance_sampling import importance_sample_log_likelihood
import os
from utils import SimpleLogger, load_latest_model


class VAERunner:
    def __init__(self, args, config):
        self.args = args
        self.config = config
        if args.device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            self.device = device
        else:
            self.device = args.device
        self.log_path = os.path.join(args.exp, args.doc)
        self.local_logger = SimpleLogger(self.log_path)

    def train(self):
        # Log args and config with local logger
        self.local_logger.log(str(self.args))
        self.local_logger.log(str(self.config))
        # Log args and config with comet
        if self.args.comet:
            experiment_config = comet_ml.ExperimentConfig(
                name=self.args.doc, tags=["train", self.config.data.dataset, self.args.vae]
            )
            experiment = comet_ml.start(
                api_key=os.environ["COMET_API_KEY"],
                project_name="scalability-student-t-vae",
                workspace=os.environ["COMET_WORKSPACE"],
                experiment_config=experiment_config
            )
            experiment.log_parameters(self.config)
            experiment.log_parameters(self.args)
        # Get datasets
        if self.config.data.dataset in ["CelebAGender", "BIMCV"]:
            train_dataset, val_dataset, _, _ = get_dataset(self.args, self.config)
        elif self.config.data.dataset == "Statlog":
            train_dataset, val_dataset, _ = get_dataset(self.args, self.config)
        else:
            raise ValueError("Unknown dataset in train()")
        # Get data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            drop_last=False
        )
        # Get model
        model = VAE(self.config, self.args.vae).to(self.device)
        # Get optimizer
        optimizer = get_optimizer(self.config, model.parameters())
        # Train
        model.train()
        total_train_loss = 0
        best_val_loss = float("inf")
        best_saved_model_epoch = 0
        variable_validation_freq = self.config.training.validation_freq
        patient_cnt = 0
        for epoch in range(self.config.training.n_epochs):
            for i, (x, _) in enumerate(train_loader):
                x = x.to(self.device)
                optimizer.zero_grad()
                loss = calculate_loss(model, x)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()
            total_train_loss /= len(train_dataset)
            msg = f"Epoch {epoch}, train nelbo: {total_train_loss}"
            print(msg)

            # Logging
            self.local_logger.log(msg)
            if self.args.comet:
                experiment.log_metric("train_nelbo", total_train_loss, step=epoch)

            # Validation for early stopping
            if epoch % variable_validation_freq == 0 and epoch > 0:
                # Get validation data loader
                val_loader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=self.config.validation.batch_size,
                    shuffle=False,
                    drop_last=False
                )
                model.eval()
                total_val_loss = 0
                with torch.no_grad():
                    for i, (x, _) in enumerate(val_loader):
                        x = x.to(self.device)
                        loss = calculate_loss(model, x)
                        total_val_loss += loss.item()
                total_val_loss /= len(val_dataset)
                msg = f"Epoch {epoch}, val nelbo: {total_val_loss}"
                print(msg)

                # Logging
                self.local_logger.log(msg)
                if self.args.comet:
                    experiment.log_metric("val_nelbo", total_val_loss, step=epoch)

                if total_val_loss <= best_val_loss:
                    best_val_loss = total_val_loss
                    patient_cnt = 0
                    variable_validation_freq = self.config.training.validation_freq
                    best_saved_model_epoch = epoch
                    # Save model
                    self.local_logger.save_model_checkpoint(model, epoch)
                    if self.args.comet:
                        log_model(experiment=experiment, model=model, model_name=f"best_model_epoch_{epoch}")
                else:
                    variable_validation_freq = 1
                    if patient_cnt >= self.config.training.patience:
                        # End training
                        msg=f"Early stopping at epoch {epoch}, best saved model at epoch {best_saved_model_epoch}"
                        print(msg)
                        # Logging
                        self.local_logger.log(msg)
                        if self.args.comet:
                            experiment.log_text(msg)
                            experiment.end()
                        break
                    else:
                        patient_cnt += 1
                model.train()

    def test(self, d_shift):
        # Get model
        model = VAE(self.config, self.args.vae).to(self.device)
        model = load_latest_model(model, self.log_path)
        model.to(self.device)
        model.eval()

        # Get datasets
        if self.config.data.dataset in ["CelebAGender", "BIMCV"]:
            if d_shift:
                _, _, _, test_dataset = get_dataset(self.args, self.config)
            else:
                _, _, test_dataset, _ = get_dataset(self.args, self.config)
        elif self.config.data.dataset == "Statlog":
            if d_shift:
                raise ValueError("Statlog does not have shifted data")
            _, _, test_dataset = get_dataset(self.args, self.config)
        else:
            raise ValueError("Unknown dataset in test()")

        # Perform testing steps
        mean_log_px = importance_sample_log_likelihood(
            model, test_dataset, K=100, batch_size=self.config.testing.batch_size, device=self.device)
        print(f"Mean log likelihood estimate: {mean_log_px}")
        # Evaluate testset loss
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.config.testing.batch_size,
            shuffle=False,
            drop_last=False
        )
        total_test_loss = 0
        with torch.no_grad():
            for i, (x, _) in enumerate(test_loader):
                x = x.to(self.device)
                loss = calculate_loss(model, x)
                total_test_loss += loss.item()
        total_test_loss /= len(test_dataset)
        print(f"Test nelbo: {total_test_loss}")

    def sample(self):
        raise NotImplementedError("Sampling not implemented yet")
