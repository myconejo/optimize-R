library(torch)
library(dplyr)

# Use GPU if available
device <- if (cuda_is_available()) torch_device("cuda") else torch_device("cpu")

# Load and prepare iris data
data(iris)
iris <- iris %>%
    mutate(Species = as.integer(as.factor(Species)))  # Make Species 1-based integer

# Shuffle rows and split into train/test
set.seed(42)
torch_manual_seed(42)
n <- nrow(iris)
train_idx <- sample(1:n, size = 0.8 * n)  # 80% train
test_idx <- setdiff(1:n, train_idx)

iris_train <- iris[train_idx, ]
iris_test <- iris[test_idx, ]

# Normalize based on train stats (important!)
X_train <- scale(as.matrix(iris_train[, 1:4]))
X_test <- scale(as.matrix(iris_test[, 1:4]),
                center = attr(X_train, "scaled:center"),
                scale = attr(X_train, "scaled:scale"))

Y_train <- iris_train$Species
Y_test <- iris_test$Species

# Convert to torch tensors
device <- torch_device(if (cuda_is_available()) "cuda" else "cpu")

X_train_tensor <- torch_tensor(X_train, dtype = torch_float(), device = device)
Y_train_tensor <- torch_tensor(Y_train, dtype = torch_long(), device = device)

X_test_tensor <- torch_tensor(X_test, dtype = torch_float(), device = device)
Y_test_tensor <- torch_tensor(Y_test, dtype = torch_long(), device = device)

# Define 2-layer MLP with ReLU activation
MLP <- nn_module(
    "MLP",
    initialize = function(input_dim, hidden_dim, output_dim) {
        # model structure
        self$fc1 <- nn_linear(input_dim, hidden_dim)
        self$fc2 <- nn_linear(hidden_dim, output_dim)
        self$relu <- nn_relu()
        
        # weight initialization
        self$initialize_weights()
        
    },
    
    initialize_weights = function(seed = 42) {
        torch_manual_seed(seed)
        nn_init_kaiming_uniform_(self$fc1$weight, nonlinearity = "relu")
        nn_init_constant_(self$fc1$bias, 0)
        
        nn_init_kaiming_uniform_(self$fc2$weight, nonlinearity = "linear")
        nn_init_constant_(self$fc2$bias, 0)
    },
    
    forward = function(x) {
        x <- self$relu(self$fc1(x))
        x <- self$fc2(x)
        x
    }
)

# train_model: returns the final test loss
train_model <- function(model, optimizer, epochs, opt_name) {
    # define criterion
    criterion <- nn_cross_entropy_loss()
    # best_test_loss <- 'inf'
    
    # training starts
    for (epoch in 1:epochs) {
        x <- X_train_tensor
        y <- Y_train_tensor
        
        prev_params = list()
        global_g_dir <- NULL
        
        params <- model$named_parameters()
        for (p in seq_along(params)) {
            # Get updated parameter
            name <- names(params)[p]
            param <- params[[p]]
            
            if (param$ndim > 1) {
                temp_param <- param$detach()$transpose(1, 2)$contiguous()
                # temp_grad <- param$grad$detach()$transpose(1, 2)$contiguous()
            } else {
                temp_param <- param$detach()
                # temp_grad <- param$grad$detach()
            }
            
            # Store pre-update parameter
            prev_params[[p]] <- param$detach()$clone()
            
            # Flatten the parameter tensor
            weight_vec <- as.numeric(temp_param$view(-1)$to(device = "cpu"))
            
            # Define directory to dump
            w_dir <- paste(name, "W.txt", sep="-")
            w_dir <- paste(opt_name, w_dir, sep="-")
            w_dir <- paste("z-", w_dir, sep="")
            
            # Create dump file
            if (!file.exists(w_dir)) {
                file.create(w_dir)
            }
            
            # Define directory to dump
            g_dir <- paste(name, "G.txt", sep="-")
            g_dir <- paste(opt_name, g_dir, sep="-")
            g_dir <- paste("z-", g_dir, sep="")
            
            # Create dump file
            if (!file.exists(g_dir)) {
                file.create(g_dir)
            }
            
            cat("_ ", file = g_dir, append = TRUE)
            
            # dump weights and gradients
            cat(paste(weight_vec, collapse = " "), "\n", file = w_dir, append = TRUE)
        }
        
        train_loss <- NULL
        
        closure <- function() {
            optimizer$zero_grad()
            output <- model(x)
            loss <- criterion(output, y)
            loss$backward()
            train_loss <<- loss
            
            # dump parameters
            params <- model$named_parameters()
            
            for (p in seq_along(params)) {
                name <- names(params)[p]
                param <- params[[p]]

                if (param$ndim > 1) {
                    temp_grad <- param$grad$detach()$transpose(1, 2)$contiguous()
                } else {
                    temp_grad <- param$grad$detach()
                }
                
                # Flatten the parameter tensor
                grad_vec <- if (!is.null(param$grad)) as.numeric(temp_grad$view(-1)$to(device = "cpu")) else NULL
                
                # Define directory to dump
                g_dir <- paste(name, "G.txt", sep="-")
                g_dir <- paste(opt_name, g_dir, sep="-")
                g_dir <- paste("Z-", g_dir, sep="")
                
                # Create dump file
                if (!file.exists(g_dir)) {
                    file.create(g_dir)
                }
                
                # Dump weights and gradients
                if (!is.null(grad_vec)) {
                    cat(paste(grad_vec, collapse = " "), "\n", file = g_dir, append = TRUE)
                }
            }

            loss
        }
        
        # Update parameters
        optimizer$step(closure)
        
        # Compute parameter delta
        params <- model$named_parameters()
        for (p in seq_along(params)) {
            # Get updated parameter
            name <- names(params)[p]
            param <- params[[p]]
            
            # Flatten the parameter tensor
            delta <- param - prev_params[[p]]
            if (delta$ndim > 1) {
                delta <- delta$transpose(1,2)$contiguous()
            }
            delta_vec <- as.numeric(delta$view(-1)$to(device = "cpu"))
            
            # Define directory to dump
            d_dir <- paste(name, "W_del.txt", sep="-")
            d_dir <- paste(opt_name, d_dir, sep="-")
            d_dir <- paste("z-", d_dir, sep="")
            
            # Create dump file
            if (!file.exists(d_dir)) {
                file.create(d_dir)
            }
            
            # dump weights and gradients
            cat(paste(delta_vec, collapse = " "), "\n", file = d_dir, append = TRUE)
        }
        
        # Evaluation
        with_no_grad({
            output <- model(X_test_tensor)
            test_loss <- criterion(output, Y_test_tensor)
        })
        
        cat(sprintf("Epoch %d || Train Loss %f || Test Loss %f\n", epoch, train_loss$item(), test_loss$item()))
    }
    # return the test loss
    return(test_loss)
}

################################################################################
# Main Routine
# Hyperparameters
epochs <- 50
opt_names = c("GD", "MOMENTUM", "LBFGS")

for (opt_name in opt_names) {
    # Instantiate model
    model <- NULL
    model <- MLP(input_dim = 4, hidden_dim = 2, output_dim = 3)
    model$to(device = device)
    model$initialize_weights()
    
    if (opt_name == "GD") {
        optimizer = optim_sgd(model$parameters, lr = 0.1)
        cat("(i) Optimizer: Gradient Descent, HP: (lr = 0.1)\n")
    }
    else if (opt_name == "MOMENTUM") {
        optimizer = optim_sgd(model$parameters, lr = 0.1, momentum = 0.9, nesterov = FALSE)
        cat("(i) Optimizer: Momentum, HP: (lr = 0.1, momentum = 0.9)\n")
    }
    else if (opt_name == "NAG") {
        optimizer = optim_sgd(model$parameters, lr = 0.1, momentum = 0.9, nesterov = TRUE)
        cat("(i) Optimizer: Nesterov Accelerated, HP: (lr = 0.1, momentum = 0.9)\n")
    }
    else if (opt_name == "LBFGS") {
        optimizer <- optim_lbfgs(model$parameters, lr = 0.1, max_iter = 20)
        cat("(i) Optimizer: L-BFGS, HP: (lr = 0.1, m = 20)\n")
    }
    
    test_loss <- train_model(model = model, optimizer = optimizer, epochs = epochs, opt_name = opt_name)

    cat("=====================================================================\n\n")
}



