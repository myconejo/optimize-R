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

# Create datasets and dataloaders
train_dataset <- tensor_dataset(X_train_tensor, Y_train_tensor)
test_dataset <- tensor_dataset(X_test_tensor, Y_test_tensor)

train_loader <- dataloader(train_dataset, batch_size = 16, shuffle = TRUE)
test_loader <- dataloader(test_dataset, batch_size = 16, shuffle = FALSE)

# Define 2-layer MLP with sigmoid activation
net <- nn_module(
    "MLP",
    initialize = function(input_dim, hidden_dim, output_dim) {
        self$fc1 <- nn_linear(input_dim, hidden_dim)
        self$fc2 <- nn_linear(hidden_dim, output_dim)
        self$sigmoid <- nn_sigmoid()
    },
    forward = function(x) {
        x <- self$sigmoid(self$fc1(x))
        x <- self$fc2(x)
        x
    }
)

# Instantiate model
model <- net(input_dim = 4, hidden_dim = 2, output_dim = 3)
model$to(device = device)

# Loss and optimizer
criterion <- nn_cross_entropy_loss()
opt_name = "SGD"
if (opt_name == "Momentum") {
    optimizer <- optim_sgd(model$parameters, lr=0.1, momentum = 0.9)
} else {
    opt_name = "SGD"
    optimizer <- optim_sgd(model$parameters, lr = 0.1)
}


# Training loop
library(coro)

epochs <- 50
write_date <- TRUE
time <- Sys.time()

for (epoch in 1:epochs) {
    # first step: indicates the first step within the epoch 
    first_step <- TRUE
    
    # cat(sprintf("Epoch-%d\n", epoch))
    
    # SGD
    train_loss <- 0
    train_data <- 0
    coro::loop(for (batch in train_loader) {
        optimizer$zero_grad()
        
        x <- batch[[1]]
        y <- batch[[2]]
        
        output <- model(x)
        loss <- criterion(output, y)
        
        loss$backward()
        train_data <- train_data + y$size()[[1]]
        train_loss <- train_loss + loss$item() * y$size()[[1]] 
        
        # Store copies of parameters before the update
        prev_params <- list()
        params <- model$named_parameters()
        
        for (i in seq_along(params)) {
            name <- names(params)[i]
            param <- params[[i]]
            
            # Clone the full tensor before optimizer step
            prev_params[[i]] <- param$detach()$clone()
            
            if (param$ndim > 1) {
                temp_param <- param$detach()$transpose(1, 2)$contiguous()
                temp_grad <- param$grad$detach()$transpose(1, 2)$contiguous()
            } else {
                temp_param <- param$detach()
                temp_grad <- param$grad$detach()
            }
            
            # Save current parameter values
            param_vec <- as.numeric(temp_param$view(-1)$to(device = "cpu"))
            grad_vec <- if (!is.null(param$grad)) as.numeric(temp_grad$view(-1)$to(device = "cpu")) else NULL
            
            pf <- paste(paste(opt_name, name, sep = "-"), "P.txt", sep = "-")
            gf <- paste(paste(opt_name, name, sep = "-"), "GRAD.txt", sep = "-")
            
            if (write_date) {
                cat(opt_name, "\n", file = pf, append = TRUE)
                cat(opt_name, "\n", file = gf, append = TRUE)
            }
            
            if (first_step && FALSE) {
                cat(sprintf("\nEpoch-%d\n", epoch), file = pf, append = TRUE)
                cat(sprintf("\nEpoch-%d\n", epoch), file = gf, append = TRUE)
            }
            
            cat(paste(param_vec, collapse = " "), "\n", file = pf, append = TRUE)
            
            if (!is.null(grad_vec)) {
                cat(paste(grad_vec, collapse = " "), "\n", file = gf, append = TRUE)
            } else {
                cat("NULL\n", file = gf, append = TRUE)
            }
        }
        
        write_date <- FALSE
        first_step <- FALSE
        
        optimizer$step()
        
        # Compute and write parameter deltas
        params <- model$named_parameters()
        for (i in seq_along(params)) {
            name <- names(params)[i]
            param <- params[[i]]
            
            delta <- param - prev_params[[i]]
            if (delta$ndim > 1) {
                delta <- delta$transpose(1,2)$contiguous()
            }
            delta_vec <- as.numeric(delta$view(-1)$to(device = "cpu"))
            
            df <- paste(paste(opt_name, name, sep = "-"), "Pdel.txt", sep = "-")
            cat(paste(delta_vec, collapse = " "), "\n", file = df, append = TRUE)
        }
    })
    
    test_loss <- 0 
    test_data <- 0
    coro::loop(for (batch in test_loader) {
        x <- batch[[1]]
        y <- batch[[2]]
        
        output <- model(x)
        loss <- criterion(output, y)
        
        test_loss <- test_loss + loss * y$size()[[1]]
        test_data <- test_data + y$size()[[1]]
    })
    
    train_loss <- train_loss / train_data
    test_loss <- test_loss / test_data
    
    cat(sprintf("Epoch %d || Train Loss %f || Test Loss %f\n", epoch, train_loss, test_loss))
    
}

