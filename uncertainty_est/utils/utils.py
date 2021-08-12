import torch
import numpy as np
from tqdm import tqdm
from scipy.integrate import trapz


def to_np(tensor: torch.Tensor):
    return tensor.detach().cpu().numpy()


def eval_func_on_grid(
    density_func,
    interval=(-10, 10),
    num_samples=200,
    device="cpu",
    dimensions=2,
    batch_size=10_000,
    dtype=torch.float32,
):
    interp = torch.linspace(*interval, num_samples)
    grid_coords = torch.meshgrid(*[interp for _ in range(dimensions)])
    grid = torch.stack([coords.reshape(-1) for coords in grid_coords], 1).to(dtype)

    vals = []
    for samples in tqdm(torch.split(grid, batch_size)):
        vals.append(density_func(samples.to(device)).cpu())
    vals = torch.cat(vals)
    return grid_coords, vals


def estimate_normalizing_constant(
    density_func,
    interval=(-10, 10),
    num_samples=200,
    device="cpu",
    dimensions=2,
    batch_size=10_000,
    dtype=torch.float32,
):
    """
    Numerically integrate a funtion in the specified interval.
    """
    with torch.no_grad():
        _, p_x = eval_func_on_grid(
            density_func, interval, num_samples, device, dimensions, batch_size, dtype
        )

        dx = (abs(interval[0]) + abs(interval[1])) / num_samples
        # Integrate one dimension after another
        grid_vals = to_np(p_x).reshape(*[num_samples for _ in range(dimensions)])
        for _ in range(dimensions):
            grid_vals = trapz(grid_vals, dx=dx, axis=-1)

        return torch.tensor(grid_vals)


def sum_except_batch(x, num_batch_dims=1):
    """Sums all elements of `x` except for the first `num_batch_dims` dimensions."""
    if x.ndimension() == 1:
        return x
    reduce_dims = list(range(num_batch_dims, x.ndimension()))
    return torch.sum(x, dim=reduce_dims)


def split_leading_dim(x, shape):
    """Reshapes the leading dim of `x` to have the given shape."""
    new_shape = torch.Size(shape) + x.shape[1:]
    return torch.reshape(x, new_shape)


def inverse_normalize(tensor, mean, std, inplace=False):
    if not inplace:
        tensor = tensor.clone()

    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)

    return tensor


def bold_best(to_bold, numeric_df=None, order="max"):
    if numeric_df is None:
        numeric_df = to_bold

    if isinstance(order, str):
        order = [
            order,
        ] * len(to_bold.columns)

    for k, col_order in zip(to_bold.columns, order):
        float_series = numeric_df[k].astype(float)
        if col_order == "max":
            max = float_series == float_series.max()
        elif col_order == "min":
            max = float_series == float_series.min()
        else:
            raise ValueError("Ordering not supported.")
        max_idxs = max[max].index.values
        for max_idx in max_idxs:
            to_bold.loc[max_idx, k] = f"\\bfseries{{{to_bold.loc[max_idx, k]}}}"


def pandas_to_latex(df_table, index=True, multicolumn=False, **kwargs) -> None:
    latex = df_table.to_latex(multicolumn=multicolumn, index=index, **kwargs)

    if multicolumn:
        latex_lines = latex.splitlines()

        insert_line_counter = 0
        for j, level in enumerate(df_table.columns.levels[:-1]):
            midrule_str = ""
            codes = np.array(df_table.columns.codes[j])
            indices = np.nonzero(codes[:-1] != codes[1:])[0]

            if index:
                indices += df_table.index.nlevels + 1
                n_columns = len(codes) + df_table.index.nlevels

            indices = (
                np.array([df_table.index.nlevels] + indices.tolist() + [n_columns]) + 1
            )
            for start, end in zip(indices[:-1], indices[1:]):
                midrule_str += f"\cmidrule(l){{{start}-{end - 1}}} "

            latex_lines.insert(3 + insert_line_counter, midrule_str)
            insert_line_counter += j + 2
        latex = "\n".join(latex_lines)

    return latex


if __name__ == "__main__":
    dims = 2
    samples = 100
    print(
        estimate_normalizing_constant(
            lambda x: torch.empty(x.shape[0]).fill_(1 / (samples ** dims)),
            num_samples=samples,
            dimensions=dims,
        )
    )
