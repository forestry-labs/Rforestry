suppressMessages(library(Rforestry))

stdout <- stdout()
test_stdout <- function(testName, results) {
    write(paste0("=== ", testName, " ==="), stdout)
    results <- round(results, 6)
    for (res in results) {
        if (is.nan(res)) {
            write("nan", stdout)
        } else{
            write(sprintf("%.6f", res), stdout)
        }
    }
}

x <- iris [, -c(1, 5)]
y <- iris[, 1]

# Test default parameters
forest <- forestry(x, y, scale = FALSE, seed = 2)
pred_avg <- predict(forest, x, aggregation = "average")
pred_oob <- predict(forest, x, aggregation = "oob")

test_stdout("Default parameters: average", pred_avg)
test_stdout("Default parameters: oob", pred_oob)

# Test oob honest
forest <- forestry(x, y, scale = FALSE, seed = 2, OOBhonest = TRUE)
pred_avg <- predict(forest, x, aggregation = "average")
pred_oob <- predict(forest, x, aggregation = "oob")

test_stdout("OOB honest: average", pred_avg)
test_stdout("OOB honest: oob", pred_oob)

# Test groups
groups = as.factor(rep(1:(nrow(x) / 10), each = 10))
forest <- forestry(x, y, scale = FALSE, seed = 2, groups = groups)
pred_avg <- predict(forest, x, aggregation = "average")
pred_oob <- predict(forest, x, aggregation = "oob")

test_stdout("Groups: average", pred_avg)
test_stdout("Groups: oob", pred_oob)

# Test oob honest and groups
forest <- forestry(x, y, scale = FALSE, seed = 2, OOBhonest = TRUE, groups = groups)
pred_avg <- predict(forest, x, aggregation = "average")
pred_oob <- predict(forest, x, aggregation = "oob")

test_stdout("OOB honest and groups: average", pred_avg)
test_stdout("OOB honest and groups: oob", pred_oob)
