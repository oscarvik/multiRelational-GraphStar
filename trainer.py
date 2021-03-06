import time
import torch
from os import path, mkdir
import utils.tensorboard_writer as tw
import utils.gsn_argparse as gap
from utils.gsn_argparse import tab_printer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from module.graph_star import GraphStar


def get_edge_info(data, type):
    attr = type + "_pos_edge_index"
    edge_index = getattr(data, attr) if hasattr(data, attr) else data.edge_index

    attr = type + "_edge_type"
    edge_type = getattr(data, attr) if hasattr(data, attr) else data.edge_type

    # Originally list of zeroes, now list of labelencoded relationships
    return edge_index, edge_type


def train_transductive(
    model,
    optimizer,
    data,
    device,
    mode="train",
    cal_mrr_score=False,
):
    lp_auc = lp_ap = None
    if mode == "train":
        model.train()
    else:
        model.eval()
    # appearently comsumes memory
    # data.to(device)

    optimizer.zero_grad()

    train_edge_index, train_edge_type = get_edge_info(data, "train")
    star_seed = data.star if hasattr(data, "star") else None

    if mode == "train":
        logits_lp, logits_star = model(
            data.x,
            train_edge_index,
            data.batch,
            star=star_seed,
            edge_type=train_edge_type,
        )
    else:
        with torch.no_grad():
            logits_lp, logits_star = model(
                data.x,
                train_edge_index,
                data.batch,
                star=star_seed,
                edge_type=train_edge_type,
            )

    pei, pet = get_edge_info(data, mode)
    if mode == "train":
        nei = data.train_neg_edge_index
    elif mode == "val":
        nei = data.val_neg_edge_index
    else:
        nei = data.test_neg_edge_index

    net = torch.randint(
        low=min(data.edge_type), high=max(data.edge_type), size=(nei.size(-1),)
    )
    nei, net = nei.to(pei.device), net.to(pei.device)
    ei = torch.cat([pei, nei], dim=-1)
    et = torch.cat([pet, net], dim=-1)
    # TODO: Need to save logits, edge index and edge type
    model.updateZ(logits_lp)

    pred = model.lp_score(logits_lp, ei, et)
    y = torch.cat(
        [logits_lp.new_ones(pei.size(-1)), logits_lp.new_zeros(nei.size(-1))], dim=0
    )

    loss = model.lp_loss(pred, y)
    lp_auc, lp_ap = model.lp_test(pred, y)

    if ((mode == "test")) and cal_mrr_score:
        model.lp_log_ranks(logits_lp, pei, pet, data.edge_index, data.edge_type)
    if mode == "train":
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
        optimizer.step()

    return loss, lp_auc, lp_ap


def trainer(
    args,
    DATASET_NAME,
    dataset,
    relation_embeddings,
    num_relations,
    num_features=0,
    epochs_per_test=1,
    epoch_per_val=1,
    num_epoch=200,
    save_per_epoch=100,
    cal_mrr_score=True,
):

    # GPU cuDNN auto tuner
    # torch.backends.cudnn.benchmark = True
    print(f"torch: {torch.__version__} \n")
    tab_printer(args)

    model = GraphStar(
        num_features=num_features,
        relation_embeddings=relation_embeddings,
        hid=args.hidden,
        num_star=args.num_star,
        star_init_method=args.star_init_method,
        heads=args.heads,
        cross_star=args.cross_star,
        num_layers=args.num_layers,
        cross_layer=args.cross_layer,
        dropout=args.dropout,
        coef_dropout=args.coef_dropout,
        residual=args.residual,
        residual_star=args.residual_star,
        layer_norm=args.layer_norm,
        activation=args.activation,
        layer_norm_star=args.layer_norm_star,
        use_e=args.use_e,
        num_relations=num_relations,
        one_hot_node=args.one_hot_node,
        one_hot_node_num=args.one_hot_node_num,
        relation_score_function=args.relation_score_function,
        additional_self_loop_relation_type=args.additional_self_loop_relation_type,
        additional_node_to_star_relation_type=args.additional_node_to_star_relation_type,
    )

    model.to(args.device)
    dataset.to(args.device)
    tw.init_writer(DATASET_NAME)
    tw.write_text("model/info", gap.args2string(args, sort_dict=True))

    # Create directory, if it doesn't already exists
    out_path = "output"
    if not path.exists(out_path):
        mkdir(out_path)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    scheduler = ReduceLROnPlateau(
        optimizer,
        "min",
        patience=args.patience,
        verbose=True,
        factor=0.5,
        cooldown=30,
        min_lr=args.lr / 100,
    )
    max_test_lp = 0
    max_val_lp = 0

    for epoch in range(1, num_epoch + 1):
        print("\n=================== Epoch: {:02d} ===================\n".format(epoch))
        start = time.time()
        cal_mrr_score = (
            epoch == num_epoch or epoch % save_per_epoch == 0
        )  # only test rank each time model is saved (per 100th epoch) and on last epoch
        train_loss, train_lp_auc, train_lp_ap = train_transductive(
            model,
            optimizer,
            dataset,
            args.device,
            mode="train",
        )
        if epoch % epoch_per_val == 0:  # TODO: undo break
            val_loss, val_lp_auc, val_lp_ap = train_transductive(
                model,
                optimizer,
                dataset,
                args.device,
                mode="val",
            )
        else:
            val_loss, val_lp_auc, val_lp_ap = 0, 0, 0

        if epoch % epochs_per_test == 0:  # TODO undo break
            test_loss, test_lp_auc, test_lp_ap = train_transductive(
                model,
                optimizer,
                dataset,
                args.device,
                mode="test",
                cal_mrr_score=cal_mrr_score,
            )
        else:
            test_loss, test_lp_auc, test_lp_ap = 0, 0, 0

        max_test_lp = max((test_lp_ap + test_lp_auc) / 2, max_test_lp)
        max_val_lp = max((val_lp_ap + val_lp_auc) / 2, max_val_lp)

        tw.log_epoch(
            DATASET_NAME,
            train_lp_auc,
            train_lp_ap,
            train_loss,
            val_lp_auc,
            val_lp_ap,
            val_loss,
            test_lp_auc,
            test_lp_ap,
            test_loss,
            max_test_lp,
            max_val_lp,
        )
        print("Epoch duration: {:.1f}".format((time.time() - start)))
        scheduler.step(train_loss)
        scheduler.step(val_loss)
        scheduler.step(test_loss)

        if epoch % save_per_epoch == 0:
            torch.save(model, path.join(out_path, DATASET_NAME + ".pkl"))

    tw.writer.close()
