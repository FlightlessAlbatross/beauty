library(ggplot2)
setwd("H:/MyDocuments/Granular/beauty")


read_logfile <- function(path){
# Load the text data
lines <- readLines(path) # Replace with the actual path to your .txt file

# Parse the lines into a data.frame
results <- do.call(rbind, lapply(lines, function(line) {
  # Extract the model information and metrics
  model_info <- sub("Model: (.*?),.*", "\\1", line)
  metrics <- sub(".*Accuracy: ([0-9.]+), F1: ([0-9.]+), Kendalls Tau: ([0-9.]+)", "\\1,\\2,\\3", line)
  
  # Split the model information into components
  model_parts <- unlist(strsplit(model_info, "__"))
  
  # Extract the metrics as numeric values
  metric_values <- as.numeric(unlist(strsplit(metrics, ",")))
  
  # Combine into a single row
  c(model_parts, metric_values)
}))

# Convert to a data.frame and name the columns
results_df <- as.data.frame(results, stringsAsFactors = FALSE)
colnames(results_df) <- c("country", "target_variable", "sampling_method", "model_type", "class_balance", "sugar", 
                          "accuracy", "F1", "KendallsTau")

# Convert the metrics to numeric
results_df$accuracy <- as.numeric(results_df$accuracy)
results_df$F1 <- as.numeric(results_df$F1)
results_df$KendallsTau <- as.numeric(results_df$KendallsTau)
results_df
}


keep_cols <- c('target_variable', 'model_type', 'class_balance', 'KendallsTau', 'Testcountry')

outcountry_test <- read_logfile("data/models/validation_logfile.txt")
outcountry_test$Testcountry <- "UK"
table(outcountry_test$target_variable)

incountry_test <- read_logfile("data/models/logfile.txt")
incountry_test$Testcountry <- "DE"
incountry_test <- incountry_test[incountry_test$country == 'DE' &  incountry_test$sampling_method == 'all_pixels' ,]
table(incountry_test$target_variable)

results <- rbind(outcountry_test, 
                 incountry_test)

forplot <- rbind(outcountry_test[,keep_cols], 
                 incountry_test[,keep_cols])

# decide model type
ggplot(forplot[forplot$class_balance == 'asis',]) + geom_bar(aes(x = Testcountry, y = KendallsTau, fill = model_type), stat = 'identity', position = 'dodge') + theme_bw()
ggsave('data/models/__plots/model_class.png')


ggplot(forplot) + geom_bar(aes(x = Testcountry, y = KendallsTau, fill = model_type), stat = 'identity', position = 'dodge') + theme_bw()


# we use XGB
# what is better, unique or beauty
ggplot(forplot[forplot$class_balance == 'asis', ]) +
  geom_bar(aes(x = Testcountry, y = KendallsTau, fill = target_variable), stat = 'identity', position = 'dodge') + theme_bw() + 
  facet_grid(model_type .~)
ggsave('data/models/__plots/beautyvunique.png')


ggplot(forplot[forplot$model_type == 'XGB', ]) +
  geom_bar(aes(x = Testcountry, y = KendallsTau, fill = class_balance), stat = 'identity', position = 'dodge') + theme_bw()



ggplot(outcountry_test) + geom_bar(aes(x = model_type, y = KendallsTau, fill = class_balance), stat = 'identity', position = 'dodge') + 
  facet_grid(target_variable ~ .) + theme_bw()
ggsave('data/models/__plots/sampling.png')