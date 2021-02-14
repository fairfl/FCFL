
from utils import getNumParams

def train(args):

    print("Preference Vector = {}".format(preference))

    dataset = args.dataset


    # LOAD DATASET
    # ------------
    # MultiMNIST: multi_mnist.pickle
    if dataset == 'emotion':
        with open('data/emotion.pkl', 'rb') as f:
            trainX, trainLabel, testX, testLabel = pickle.load(f)

    trainX = torch.from_numpy(trainX).float()
    trainLabel = torch.from_numpy(trainLabel).float()
    testX = torch.from_numpy(testX).float()
    testLabel = torch.from_numpy(testLabel).float()
    n_tasks = testLabel.shape[1]
    n_feats = testX.shape[1]

    train_set = torch.utils.data.TensorDataset(trainX, trainLabel)
    test_set = torch.utils.data.TensorDataset(testX, testLabel)

    batch_size = 50
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=False)

    print('==>>> total trainning batch number: {}'.format(len(train_loader)))
    print('==>>> total testing batch number: {}'.format(len(test_loader)))
    # ---------***---------

    # DEFINE MODEL
    # ---------------------
    model = RegressionTrain(RegressionModel(n_feats, n_tasks))
    _, n_params = getNumParams(model.parameters())
    print(f"# params={n_params}; # layers={len(model.model.layers)}")

    if args.load
    # model.randomize()
    if torch.cuda.is_available():
        model.cuda()
    # ---------***---------

    # DEFINE OPTIMIZERS
    # -----------------
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     optimizer, milestones=[15, 30, 45, 60, 75, 90], gamma=0.8)

    # Instantia EPO Linear Program Solver
    epo_lp = EPO_LP(m=n_tasks, n=n_params, r=preference)
    # ---------***---------

    # CONTAINERS FOR KEEPING TRACK OF PROGRESS
    # ----------------------------------------
    task_train_losses = []
    train_accs = []
    # ---------***---------

    # TRAIN
    # -----
    for t in range(niter):
        # scheduler.step()

        n_linscalar_adjusts = 0
        descent = 0.
        model.train()
        for (it, batch) in enumerate(train_loader):
            X = batch[0]
            ts = batch[1]
            if torch.cuda.is_available():
                X = X.cuda()
                ts = ts.cuda()

            # Obtain losses and gradients
            grads = {}
            losses = []
            for i in range(n_tasks):
                optimizer.zero_grad()
                task_loss = model(X, ts)
                losses.append(task_loss[i].data.cpu().numpy())
                task_loss[i].backward()

                # One can use scalable method proposed in the MOO-MTL paper 
                # for large scale problem; but we use the gradient
                # of all parameters in this experiment.
                grads[i] = []
                for param in model.parameters():
                    if param.grad is not None:
                        grads[i].append(Variable(param.grad.data.clone().flatten(), requires_grad=False))

            grads_list = [torch.cat(grads[i]) for i in range(len(grads))]
            G = torch.stack(grads_list)
            GG = G @ G.T
            losses = np.stack(losses)

            try:
                # Calculate the alphas from the LP solver
                alpha = epo_lp.get_alpha(losses, G=GG.cpu().numpy(), C=True)
                if epo_lp.last_move == "dom":
                    descent += 1
            except Exception as e:
                # print(e)
                # print(f'losses:{losses}')
                # print(f'C:\n{GG.cpu().numpy()}')
                # raise RuntimeError('manual tweak')
                alpha = None
            if alpha is None:   # A patch for the issue in cvxpy
                alpha = preference / preference.sum()
                n_linscalar_adjusts += 1

            if torch.cuda.is_available:
                alpha = n_tasks * torch.from_numpy(alpha).cuda()
            else:
                alpha = n_tasks * torch.from_numpy(alpha)
            # Optimization step
            optimizer.zero_grad()
            task_losses = model(X, ts)
            weighted_loss = torch.sum(task_losses * alpha)  # * 5. * max(epo_lp.mu_rl, 0.2)
            weighted_loss.backward()
            optimizer.step()

        print(f"\tdescent={descent/len(train_loader)}")
        if n_linscalar_adjusts > 0:
            print(f"\t # linscalar steps={n_linscalar_adjusts}")

        # Calculate and record performance
        if t == 0 or (t + 1) % 2 == 0:
            model.eval()
            with torch.no_grad():
                total_train_loss = []

                for (it, batch) in enumerate(test_loader):

                    X = batch[0]
                    ts = batch[1]
                    if torch.cuda.is_available():
                        X = X.cuda()
                        ts = ts.cuda()

                    valid_train_loss = model(X, ts)
                    total_train_loss.append(valid_train_loss)

                total_train_loss = torch.stack(total_train_loss)
                average_train_loss = torch.mean(total_train_loss, dim=0)

            # record and print
            if torch.cuda.is_available():

                task_train_losses.append(average_train_loss.data.cpu().numpy())

                print('{}/{}: train_loss={}'.format(
                    t + 1, niter, task_train_losses[-1]))

    # torch.save(model.model.state_dict(),
    #            f'./saved_model/{dataset}_{base_model}_niter_{niter}.pickle')

    result = {"training_losses": task_train_losses}

    return result