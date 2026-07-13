// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
module neural_networks;

__include basetypes.ireal;
__include basetypes.imodel;
__include basetypes.identity;
__include components.conversions;
__include debug.model_recorder;
__include components.model_chain;
__include components.linear_layer;
__include components.activations;
__include optimizers.ioptimizer;
__include optimizers.regularized_adam_optimizer;
__include optimizers.full_precision_optimizer;
__include optimizers.deterministic_optimizer;
__include losses.losses;
