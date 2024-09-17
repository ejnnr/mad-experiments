import cupbearer as cup
import lightning as L
import torch
import transformers
from torchmetrics.classification import Accuracy


class BackdooredClassifier(L.LightningModule):
    def __init__(
        self,
        model: transformers.AutoModelForSequenceClassification,
        lr: float,
        num_classes: int | None = None,
        num_labels: int | None = None,
        val_loader_names: list[str] | None = None,
        test_loader_names: list[str] | None = None,
        save_hparams: bool = True,
        task: str = "multiclass",
        weight_decay: float = 0.0,
    ):
        super().__init__()

        if save_hparams:
            self.save_hyperparameters(ignore=["model"])
        if val_loader_names is None:
            val_loader_names = []
        if test_loader_names is None:
            test_loader_names = []

        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.val_loader_names = val_loader_names
        self.test_loader_names = test_loader_names
        self.task = task
        self.loss_func = self._get_loss_func(self.task)
        self.train_accuracy = Accuracy(
            task=self.task, num_classes=num_classes, num_labels=num_labels
        )
        self.val_accuracy = torch.nn.ModuleList(
            [
                Accuracy(task=self.task, num_classes=num_classes, num_labels=num_labels)
                for _ in val_loader_names
            ]
        )
        self.test_accuracy = torch.nn.ModuleList(
            [
                Accuracy(task=self.task, num_classes=num_classes, num_labels=num_labels)
                for _ in test_loader_names
            ]
        )

    def _get_loss_func(self, task):
        if task == "multiclass":
            return torch.nn.functional.cross_entropy
        return torch.nn.functional.binary_cross_entropy_with_logits

    def compute_task_loss(self, batch):
        (x, y), is_backdoored = batch
        output = self.model(x)
        assert output.logits.ndim == 2
        assert y.ndim == 1
        assert output.logits.shape[0] == y.shape[0]
        loss = self.loss_func(output.logits, y)
        return loss, output.logits, y

    def training_step(self, batch, batch_idx):
        loss, logits, y = self.compute_task_loss(batch)

        self.log("train/task_loss", loss, prog_bar=True)
        self.train_accuracy(logits, y)
        self.log("train/acc_step", self.train_accuracy)

        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss, logits, y = self.compute_task_loss(batch)
        name = self.test_loader_names[dataloader_idx]
        self.log(f"{name}/loss", loss)
        self.test_accuracy[dataloader_idx](logits, y)
        self.log(f"{name}/acc_step", self.test_accuracy[dataloader_idx])

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, logits, y = self.compute_task_loss(batch)
        name = self.val_loader_names[dataloader_idx]
        self.log(f"{name}/loss", loss)
        self.val_accuracy[dataloader_idx](logits, y)
        self.log(f"{name}/acc_step", self.val_accuracy[dataloader_idx])

    def on_train_epoch_end(self):
        self.log("train/acc_epoch", self.train_accuracy)

    def on_test_epoch_end(self):
        for i, name in enumerate(self.test_loader_names):
            self.log(f"{name}/acc_epoch", self.test_accuracy[i])

    def on_validation_epoch_end(self):
        for i, name in enumerate(self.val_loader_names):
            self.log(f"{name}/acc_epoch", self.val_accuracy[i])

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )


class SubtleBackdooredClassifier(L.LightningModule):
    def __init__(
        self,
        model: transformers.AutoModelForSequenceClassification,
        detector: cup.detectors.AnomalyDetector,
        regularization_names: list[str],
        lr: float,
        detector_lr: float | None = None,
        regularization_weight: float = 1.0,
        num_classes: int | None = None,
        num_labels: int | None = None,
        val_loader_names: list[str] | None = None,
        test_loader_names: list[str] | None = None,
        save_hparams: bool = True,
        task: str = "multiclass",
        weight_decay: float = 0.0,
    ):
        super().__init__()

        if save_hparams:
            self.save_hyperparameters(ignore=["model", "detector"])
        if val_loader_names is None:
            val_loader_names = []
        if test_loader_names is None:
            test_loader_names = []

        self.model = model
        self.detector = detector
        self.detector.set_model(model)
        self.regularization_names = regularization_names
        assert len(regularization_names) > 0
        self.lr = lr
        self.detector_lr = detector_lr or lr
        self.weight_decay = weight_decay
        self.regularization_weight = regularization_weight
        self.val_loader_names = val_loader_names
        self.test_loader_names = test_loader_names
        self.task = task
        self.loss_func = self._get_loss_func(self.task)
        self.train_accuracy = Accuracy(
            task=self.task, num_classes=num_classes, num_labels=num_labels
        )
        self.val_accuracy = torch.nn.ModuleList(
            [
                Accuracy(task=self.task, num_classes=num_classes, num_labels=num_labels)
                for _ in val_loader_names
            ]
        )
        self.test_accuracy = torch.nn.ModuleList(
            [
                Accuracy(task=self.task, num_classes=num_classes, num_labels=num_labels)
                for _ in test_loader_names
            ]
        )

        self.automatic_optimization = False
        self._initialized_detector_variables = False

    def _get_loss_func(self, task):
        if task == "multiclass":
            return torch.nn.functional.cross_entropy
        return torch.nn.functional.binary_cross_entropy_with_logits

    def compute_task_loss(self, batch):
        (x, y), is_backdoored = batch
        output = self.model(x)
        loss = self.loss_func(output.logits, y)
        return loss, output.logits, y, is_backdoored

    def detector_step(
        self, clean_samples, clean_activations: dict[str, torch.Tensor], batch_idx: int
    ):
        if isinstance(self.detector, cup.detectors.ActivationCovarianceBasedDetector):
            # I don't think we actually need no_grad given that activations
            # are detached?
            with torch.no_grad():
                self.detector.batch_update(clean_activations, case="trusted")
                # TODO: this is extremely inefficient, it calls pinv every batch
                self.detector._finalize_training()
                return None
        elif isinstance(self.detector, cup.detectors.FeatureModelDetector):
            loss, _ = self.detector.module._shared_step(
                (clean_samples, clean_activations)
            )
            self.log("train/vae_loss", loss)
            return loss
        else:
            raise NotImplementedError(
                f"Unsupported detector type: {type(self.detector)}"
            )

    def init_detector_variables(self, sample_batch):
        if isinstance(self.detector, cup.detectors.ActivationCovarianceBasedDetector):
            self.detector._means = {}
            self.detector._Cs = {}
            self.detector._ns = {}
            self.detector.init_variables(sample_batch, case="trusted")
        elif isinstance(self.detector, cup.detectors.FeatureModelDetector):
            self.detector._setup_training(self.detector_lr)
            # HACK: add LightningModule to make sure device placement works
            self.detector_module = self.detector.module
        else:
            raise NotImplementedError(
                f"Unsupported detector type: {type(self.detector)}"
            )

        self._initialized_detector_variables = True

    def teardown(self, stage: str):
        if stage == "fit" and isinstance(
            self.detector, cup.detectors.FeatureModelDetector
        ):
            self.detector._teardown_training()

    def training_step(self, batch, batch_idx):
        (x, y), is_backdoored = batch
        activations, outputs = cup.utils.get_activations(
            x,
            model=self.model,
            names=self.regularization_names,
            return_output=True,
            no_grad=False,
        )

        assert is_backdoored.ndim == 1 and is_backdoored.shape[0] == y.shape[0]
        logits = outputs.logits
        assert logits.ndim == 2
        assert y.ndim == 1
        assert logits.shape[0] == y.shape[0]
        task_loss = self.loss_func(logits, y)

        clean_activations = {
            k: v[~is_backdoored].detach() for k, v in activations.items()
        }
        if isinstance(x, list):
            clean_x = [
                entry for entry, backdoor in zip(x, is_backdoored) if not backdoor
            ]
        elif isinstance(x, tuple):
            clean_x = tuple(
                entry for entry, backdoor in zip(x, is_backdoored) if not backdoor
            )
        else:
            clean_x = x[~is_backdoored]
        clean_y = y[~is_backdoored]
        clean_samples = (clean_x, clean_y)

        detector_loss = self.detector_step(clean_samples, clean_activations, batch_idx)
        if detector_loss is not None:
            model_opt, detector_opt = self.optimizers()

            detector_opt.zero_grad()
            self.manual_backward(detector_loss)
            detector_opt.step()
        else:
            model_opt = self.optimizers()

        scores = self.detector._compute_layerwise_scores(x, activations)
        scores = self.detector._aggregate_scores(scores)

        model_regularization_loss = torch.mean(scores[is_backdoored])

        model_loss = task_loss + self.regularization_weight * model_regularization_loss
        model_opt.zero_grad()
        self.manual_backward(model_loss)
        model_opt.step()

        self.log("train/task_loss", task_loss, prog_bar=True)
        self.log("train/model_loss", model_loss)
        self.log("train/anomaly_loss", model_regularization_loss)
        self.train_accuracy(logits, y)
        self.log("train/acc_step", self.train_accuracy)

        return model_loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss, logits, y, is_backdoored = self.compute_task_loss(batch)
        name = self.test_loader_names[dataloader_idx]
        self.log(f"{name}/loss", loss)
        self.test_accuracy[dataloader_idx](logits, y)
        self.log(f"{name}/acc_step", self.test_accuracy[dataloader_idx])

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, logits, y, is_backdoored = self.compute_task_loss(batch)
        name = self.val_loader_names[dataloader_idx]
        self.log(f"{name}/loss", loss)
        self.val_accuracy[dataloader_idx](logits, y)
        self.log(f"{name}/acc_step", self.val_accuracy[dataloader_idx])

    def on_train_epoch_end(self):
        self.log("train/acc_epoch", self.train_accuracy)

    def on_test_epoch_end(self):
        for i, name in enumerate(self.test_loader_names):
            self.log(f"{name}/acc_epoch", self.test_accuracy[i])

    def on_validation_epoch_end(self):
        for i, name in enumerate(self.val_loader_names):
            self.log(f"{name}/acc_epoch", self.val_accuracy[i])

    def configure_optimizers(self):
        if not self._initialized_detector_variables:
            # We can't call init_detector_variables ourselves because apparently
            # the training dataloader isn't available at this point.
            # But we need the detector to be initialized before we can create the
            # detector optimizer.
            raise RuntimeError(
                "Detector variables not initialized, call init_detector_variables "
                "before Trainer.fit"
            )

        model_opt = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        if isinstance(self.detector, cup.detectors.FeatureModelDetector):
            detector_opt = torch.optim.Adam(
                self.detector.module.parameters(), lr=self.detector_lr
            )
            return model_opt, detector_opt
        return model_opt
