from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    LinearLR,
    CosineAnnealingLR,
    SequentialLR
)
from oceanography_with_flowers.utils import (
    open_pactcs30,
    load_pactcs30,
    clean_pactcs30,
    tensor_to_dataset,
    plot_dataset,
)

def train(dataloader, model, loss_fn, optimizer, scheduler, mask):
    """
    Train a model on pactcs30 data.

    Args:
        dataloader (torch.utils.data.DataLoader): Dataloader for training process.
        model: Model that is supposed to be trained.
        loss_fn: Loss function to be used.
        optimizer: Optimizer to be used.
        scheduler: Scheduler to be used.
        mask: Mask to remove land points when calculating loss.
    """

    num_batches = len(dataloader)
    model.train()
    train_loss = 0
    
    for batch, (X, y) in enumerate(dataloader):
        pred = fno(X)
        v_mask = mask.expand_as(pred)
        
        loss = loss_fn(pred[v_mask], y[v_mask])
        loss.backward()
        
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        train_loss += loss.item()

    train_loss /= num_batches
    print(f"Avg loss: {train_loss:>8f} \n")

def test(dataloader, model, loss_fn, mask):
    """
    Test a model on pactcs30 data.

    Args:
        dataloader (torch.utils.data.DataLoader): Dataloader for training process.
        model: Model that is supposed to be trained.
        loss_fn: Loss function to be used.
        mask: Mask to remove land points when calculating loss.
    """

    model.eval()
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            v_mask = mask.expand_as(pred)
            
            test_loss += loss_fn(pred[v_mask], y[v_mask]).item()

    test_loss /= num_batches
    print(f"Avg loss: {test_loss:>8f} \n")
    
    feature, label = next(iter(dataloader))
    pred = model(feature)
    pred_dataset = tensor_to_dataset(pred[0], te_mean, te_std, te_data.mask_rho, te_data.eta_rho, te_data.xi_rho, te_data.lat_rho, te_data.lon_rho)
    label_dataset = tensor_to_dataset(label[0], te_mean, te_std, te_data.mask_rho, te_data.eta_rho, te_data.xi_rho, te_data.lat_rho, te_data.lon_rho)

    print(pred_dataset)
    print("")
    print(label_dataset)
    print("")
    
    # can be used to plot results
    #plot_dataset(pred_dataset)
    #plot_dataset(label_dataset)

if __name__ == "__main__":
    # TODO: use configs for this
    fno = Pactcs30Fno()

    base_lr = 1e-3
    min_lr = 1e-6
    batch_size = 32
    epochs = 5

    loss_fn = nn.L1Loss()
    optimizer = torch.optim.AdamW(fno.parameters(), lr=base_lr)

    total_steps = 100
    warmup_steps = 10
    decay_steps = total_steps - warmup_steps

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_steps
    )

    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=decay_steps,
        eta_min=min_lr
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )

    tr_data, tr_mean, tr_std = load_pactcs30([2010])
    te_data, te_mean, te_std = load_pactcs30([2011])

    tr_data = clean_pactcs30(tr_data, tr_mean, tr_std)
    te_data = clean_pactcs30(te_data, te_mean, te_std)

    tr_dataloader = DataLoader(Pactcs30Dataset(tr_data), batch_size=batch_size, shuffle=True)
    te_dataloader = DataLoader(Pactcs30Dataset(te_data), batch_size=batch_size, shuffle=True)

    mask = torch.from_numpy(tr_data.mask_rho.values).bool()

    for i in range(epochs):
        print(f"epoch: {i}")
        train(tr_dataloader, fno, loss_fn, optimizer, scheduler, mask)
        test(te_dataloader, fno, loss_fn, mask)