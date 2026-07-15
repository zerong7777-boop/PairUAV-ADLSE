import argparse
from pathlib import Path

import torch

from reloc3r.reloc3r_relpose import Reloc3rRelpose
from reloc3r.datasets import get_data_loader
from reloc3r.pairuav_metrics import write_pairuav_devval_outputs


DEFAULT_MODEL = "Reloc3rRelpose(img_size=512, output_mode='pairuav_heading_range')"


def get_args_parser():
    parser = argparse.ArgumentParser(description="evaluation code for PairUAV dev-val")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--test_dataset", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--amp", type=int, default=1, choices=[0, 1])
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--range_min", type=float, default=-132.0)
    parser.add_argument("--range_max", type=float, default=132.0)
    return parser


def build_dataset(dataset, batch_size, num_workers, test=False):
    split = ['Train', 'Test'][test]
    print('Building {} data loader for {}'.format(split, dataset))
    loader = get_data_loader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_mem=True,
        shuffle=not test,
        drop_last=not test,
    )
    print('Dataset length: ', len(loader))
    return loader


def load_checkpoint(model, checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(payload, dict) and 'model' in payload:
        state_dict = payload['model']
    elif isinstance(payload, dict) and 'state_dict' in payload:
        state_dict = payload['state_dict']
    elif isinstance(payload, dict):
        state_dict = payload
    else:
        raise TypeError(f'Unsupported checkpoint payload type: {type(payload)!r}')
    print(model.load_state_dict(state_dict, strict=False))


@torch.no_grad()
def test(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    reloc3r_relpose = eval(args.model)
    reloc3r_relpose.to(device)
    reloc3r_relpose.eval()
    load_checkpoint(reloc3r_relpose, args.checkpoint, device)

    data_loader_test = {
        dataset.split('(')[0]: build_dataset(dataset, args.batch_size, args.num_workers, test=True)
        for dataset in args.test_dataset.split('+')
    }

    heading_prediction_deg = []
    heading_target_deg = []
    range_prediction = []
    range_target = []

    for test_name, testset in data_loader_test.items():
        print('Testing {:s}'.format(test_name))
        for batch in testset:
            view1, view2 = batch
            for view in batch:
                for name in 'img camera_intrinsics camera_pose'.split():
                    if name in view:
                        view[name] = view[name].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=bool(args.amp)):
                _, pred2 = reloc3r_relpose(view1, view2)

            pred_heading = pred2['heading_vec']
            pred_range = pred2['range_value'].view(-1)
            pred_deg = torch.rad2deg(torch.atan2(pred_heading[:, 1], pred_heading[:, 0]))

            heading_prediction_deg.append(pred_deg.detach().cpu())
            range_prediction.append(pred_range.detach().cpu())
            heading_target_deg.append(torch.as_tensor(view2['heading_deg']).detach().cpu())
            range_target.append(torch.as_tensor(view2['range_value']).detach().cpu())

    heading_prediction_deg = torch.cat(heading_prediction_deg, dim=0)
    heading_target_deg = torch.cat(heading_target_deg, dim=0)
    range_prediction = torch.cat(range_prediction, dim=0)
    range_target = torch.cat(range_target, dim=0)

    outputs = write_pairuav_devval_outputs(
        heading_prediction_deg=heading_prediction_deg,
        heading_target_deg=heading_target_deg,
        range_prediction=range_prediction,
        range_target=range_target,
        output_dir=output_dir,
        range_min=args.range_min,
        range_max=args.range_max,
    )
    print('PairUAV dev-val evaluation complete.')
    print('Saved predictions to:', outputs['val_predict_output'])


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    test(args)
