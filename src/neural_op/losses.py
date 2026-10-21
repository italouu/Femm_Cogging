import torch


def mse_loss(out, y):
    return torch.mean((out - y) ** 2)


def mae_loss(out, y):
    return torch.mean(torch.abs(out - y))


class BaseLoss:
    """
    Wrapper extensível em volta de uma função de loss pura (fn) -- __call__
    só repassa pra fn (mesma assinatura de sempre, nenhuma mudança em nenhum
    step_fn de arch) e acrescenta log_epoch como PROPRIEDADE da loss: quem
    decide o que aparece no console a cada época é a própria loss, não o
    loop de treino (fit(), src/neural_op/training_utils.py).

    Por ora toda entrada de LOSS_REGISTRY usa o log_epoch padrão abaixo
    (mesmo texto que já era hardcoded em fit()). Uma loss que precise
    destacar termos próprios (ex: erro de ajuste/"data" vs penalidade
    física -- caso de graph_div_b_loss, que já calcula `fit` e `div`
    separadamente) sobrescreve log_epoch numa subclasse; nada mais no
    projeto precisa mudar pra isso (scripts/train.py já extrai
    `loss_obj.log_epoch` do objeto, antes de qualquer functools.partial).
    """

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    def log_epoch(self, ep, train_loss, test_loss, train_time_s, eval_time_s, samples_per_s):
        print(f"epoch {ep:>4d}  train {train_loss:.4e}  test {test_loss:.4e}"
              f"  [{train_time_s:.1f}s + {eval_time_s:.1f}s eval]  {samples_per_s:.0f} samp/s")


LOSS_REGISTRY: dict = {
    'mse': BaseLoss(mse_loss),
    'mae': BaseLoss(mae_loss),
}
