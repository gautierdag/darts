import torch
import numpy as np
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


def _clip(grads, max_norm):
    total_norm = 0
    for g in grads:
        param_norm = g.data.norm(2)
        total_norm += param_norm ** 2
    total_norm = total_norm ** 0.5
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for g in grads:
            g.data.mul_(clip_coef)
    return clip_coef


class Architect(object):
    def __init__(self, model, args):
        self.network_weight_decay = args.wdecay
        self.network_clip = args.clip
        self.model = model
        self.optimizer = torch.optim.Adam(
            self.model.arch_parameters(), lr=args.arch_lr, weight_decay=args.arch_wdecay
        )

    def _compute_unrolled_model(self, hidden, input, target, eta):
        loss, hidden_next = self.model._loss(hidden, input, target)
        theta = _concat(self.model.parameters()).data
        grads = torch.autograd.grad(loss, self.model.parameters())
        clip_coef = _clip(grads, self.network_clip)
        dtheta = _concat(grads).data + self.network_weight_decay * theta
        unrolled_model = self._construct_model_from_theta(theta.sub(eta, dtheta))
        return unrolled_model, clip_coef

    def step(
        self,
        hidden_train,
        input_train,
        target_train,
        hidden_valid,
        input_valid,
        target_valid,
        network_optimizer,
        unrolled,
    ):
        eta = network_optimizer.param_groups[0]["lr"]
        self.optimizer.zero_grad()
        if unrolled:
            hidden = self._backward_step_unrolled(
                hidden_train,
                input_train,
                target_train,
                hidden_valid,
                input_valid,
                target_valid,
                eta,
            )
        else:
            hidden = self._backward_step(hidden_valid, input_valid, target_valid)
        self.optimizer.step()
        return hidden, None

    def _backward_step(self, hidden, input, target):
        loss, hidden_next = self.model._loss(hidden, input, target)
        loss.backward()
        return hidden_next

    def _backward_step_unrolled(
        self,
        hidden_train,
        input_train,
        target_train,
        hidden_valid,
        input_valid,
        target_valid,
        eta,
    ):
        unrolled_model, clip_coef = self._compute_unrolled_model(
            hidden_train, input_train, target_train, eta
        )
        unrolled_loss, hidden_next = unrolled_model._loss(
            hidden_valid, input_valid, target_valid
        )

        unrolled_loss.backward()
        dalpha = [v.grad for v in unrolled_model.arch_parameters()]
        dtheta = [v.grad for v in unrolled_model.parameters()]
        _clip(dtheta, self.network_clip)
        vector = [dt.data for dt in dtheta]
        implicit_grads = self._hessian_vector_product(
            vector, hidden_train, input_train, target_train, r=1e-2
        )

        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta * clip_coef, ig.data)

        for v, g in zip(self.model.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = g.data
            else:
                v.grad.data.copy_(g.data)
        return hidden_next

    def _construct_model_from_theta(self, theta):
        model_new = self.model.new()
        model_dict = self.model.state_dict()

        params, offset = {}, 0
        for k, v in self.model.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset : offset + v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.to(device)

    def _hessian_vector_product(self, vector, hidden, input, target, r=1e-2):
        norm = torch.cat([w.view(-1) for w in vector]).norm()
        R = r / norm

        # w+ = w + R*dw`
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), vector):
                p += R * d

        loss, _ = self.model._loss(hidden, input, target)
        grads_p = torch.autograd.grad(
            loss, self.model.arch_parameters()
        )  # dalpha { L_trn(w+) }

        # w- = w - R*dw`
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), vector):
                p -= 2.0 * R * d
        loss, _ = self.model._loss(hidden, input, target)
        grads_n = torch.autograd.grad(
            loss, self.model.arch_parameters()
        )  # dalpha { L_trn(w-) }

        # recover w
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), vector):
                p += R * d

        return [(x - y) / 2.0 * R for x, y in zip(grads_p, grads_n)]


# """ Architect controls architecture of cell by computing gradients of alphas """
# import copy
# import torch


# class Architect():
#     """ Compute gradients of alphas """
#     def __init__(self, net, w_momentum, w_weight_decay):
#         """
#         Args:
#             net
#             w_momentum: weights momentum
#         """
#         self.net = net
#         self.v_net = copy.deepcopy(net)
#         self.w_momentum = w_momentum
#         self.w_weight_decay = w_weight_decay

#     def virtual_step(self, trn_X, trn_y, xi, w_optim):
#         """
#         Compute unrolled weight w' (virtual step)
#         Step process:
#         1) forward
#         2) calc loss
#         3) compute gradient (by backprop)
#         4) update gradient
#         Args:
#             xi: learning rate for virtual gradient step (same as weights lr)
#             w_optim: weights optimizer
#         """
#         # forward & calc loss
#         loss = self.net.loss(trn_X, trn_y) # L_trn(w)

#         # compute gradient
#         gradients = torch.autograd.grad(loss, self.net.weights())

#         # do virtual step (update gradient)
#         # below operations do not need gradient tracking
#         with torch.no_grad():
#             # dict key is not the value, but the pointer. So original network weight have to
#             # be iterated also.
#             for w, vw, g in zip(self.net.weights(), self.v_net.weights(), gradients):
#                 m = w_optim.state[w].get('momentum_buffer', 0.) * self.w_momentum
#                 vw.copy_(w - xi * (m + g + self.w_weight_decay*w))

#             # synchronize alphas
#             for a, va in zip(self.net.alphas(), self.v_net.alphas()):
#                 va.copy_(a)

#     def unrolled_backward(self, trn_X, trn_y, val_X, val_y, xi, w_optim):
#         """ Compute unrolled loss and backward its gradients
#         Args:
#             xi: learning rate for virtual gradient step (same as net lr)
#             w_optim: weights optimizer - for virtual step
#         """
#         # do virtual step (calc w`)
#         self.virtual_step(trn_X, trn_y, xi, w_optim)

#         # calc unrolled loss
#         loss = self.v_net.loss(val_X, val_y) # L_val(w`)

#         # compute gradient
#         v_alphas = tuple(self.v_net.alphas())
#         v_weights = tuple(self.v_net.weights())
#         v_grads = torch.autograd.grad(loss, v_alphas + v_weights)
#         dalpha = v_grads[:len(v_alphas)]
#         dw = v_grads[len(v_alphas):]

#         hessian = self.compute_hessian(dw, trn_X, trn_y)

#         # update final gradient = dalpha - xi*hessian
#         with torch.no_grad():
#             for alpha, da, h in zip(self.net.alphas(), dalpha, hessian):
#                 alpha.grad = da - xi*h

