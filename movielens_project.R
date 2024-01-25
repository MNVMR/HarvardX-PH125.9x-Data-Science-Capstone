## ======================================================================================
# Loading necessary libraries
library(tidyverse) # For data manipulation and visualization
library(caret)     # For data partitioning and modeling
library(dplyr)     # For data manipulation
library(ggplot2)   # For data visualization

# Setting a longer timeout for operations
options(timeout = 120)

# Downloading and unzipping the MovieLens dataset
dl <- "ml-10M100K.zip"
if(!file.exists(dl)) {
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)
}

# Extracting ratings data
ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file)) {
  unzip(dl, ratings_file)
}

# Extracting movies data
movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file)) {
  unzip(dl, movies_file)
}

# Reading and formatting the ratings data
ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

# Reading and formatting the movies data
movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

# Merging ratings and movies data
movielens <- left_join(ratings, movies, by = "movieId")

# Setting seed for reproducibility
set.seed(1)

# Partitioning the data into training (edx) and testing (temp) sets
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Creating a final holdout test set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Identifying the removed entries
removed <- anti_join(temp, final_holdout_test)

# Updating the training set (edx) by adding back the removed entries
edx <- rbind(edx, removed)

# Cleaning up the environment by removing unnecessary objects
rm(dl, ratings, movies, test_index, temp, movielens, removed)

#1 ======================================================================================

# Summarizing the 'edx' dataset to count the number of unique users and unique movies
edx %>% summarize(unique_users = length(unique(userId)),
                  unique_movies = length(unique(movieId)))  
                  
# Calculating the total number of missing values in each column of the 'edx' dataset
edx %>%
  summarise_all(~sum(is.na(.)))

#2 ======================================================================================

# Separating the 'genres' column into multiple rows and displaying the first few rows of the resulting dataset
edx_genres <-edx %>% separate_rows(genres, sep = "\\|")
head(edx_genres)

#3 ======================================================================================

# Define a function to process movie data
process_movie_data <- function(data, timestamp_col = "timestamp", title_col = "title", current_year = 2024) {

  # Convert the timestamp column to POSIXct format, then format it to extract only the year.
  # The column is then renamed to 'RatingYear'.
  data <- data %>%
    mutate(!!timestamp_col := as.POSIXct(!!sym(timestamp_col), origin = "1970-01-01", tz = "EST")) %>%
    mutate(!!timestamp_col := format(!!sym(timestamp_col), "%Y")) %>%
    rename(RatingYear = !!sym(timestamp_col))
  
  # Extract the year of release from the title column.
  # This assumes the year is in the last four characters of the title.
  # Then, calculate the age of the movie based on the current year.
  year_released <- as.numeric(str_sub(data[[title_col]], start = -5, end = -2))
  data <- data %>%
    mutate(yearReleased = year_released,
           MovieAge = current_year - yearReleased)  # MovieAge is the current year minus the release year
  
  return(data)  # Return the processed data
}

# Apply the function to the 'edx' data frame and display the first few rows
edx <- process_movie_data(edx)
head(edx)  # Display the first six rows of the processed data

#4 ======================================================================================

# Define a function for creating a histogram plot for a specified variable 
create_plot <- function(data, variable_name) {
  variable <- rlang::sym(variable_name)  # Convert the variable name to a symbol for tidy evaluation

  # Generate and return a histogram plot
  data %>%
    count(!!variable) %>%  # Count occurrences of each unique value in the specified variable
    ggplot(aes(n)) +  # Create a ggplot object with number of counts as the x-axis
    geom_histogram(bins = 30, color = "white") +  # Plot histogram with 30 bins and white border for each bar
    scale_x_log10() +  # Use a logarithmic scale for the x-axis to handle wide data range
    ggtitle("Movies") +  # Set the main title of the plot
    labs(subtitle = paste("Number of ratings by", variable_name),  # Set subtitle dynamically based on variable
         x = variable_name,  # Label for x-axis
         y = "Number of Ratings") +  # Label for y-axis
    theme(panel.border = element_rect(colour = "black", fill = NA))  # Customize plot theme
}

# List of variables to create plots for
variables <- c("userId", "movieId")

# Apply the create_plot function to each variable in 'variables' using the 'edx' dataset
# Store the resulting list of plots in 'plots'
plots <- lapply(variables, function(v) create_plot(edx, v))

print(plots[[1]])
print(plots[[2]])
#5 ======================================================================================

# Plotting a histogram of the 'rating' variable in the 'edx' dataset
ggplot(edx, aes(x = rating)) +  # Initialize a ggplot, setting 'rating' as the x-axis variable
  geom_histogram(binwidth = 1, color = "white") +  # Create a histogram with bins of width 1 and white borders for bars
  scale_x_continuous(breaks = 1:5, labels = 1:5) +  # Set the x-axis scale to be continuous with breaks and labels from 1 to 5
  labs(title = "Ratings Distribution",  # Add a title to the plot
       x = "Rating Awarded",  # Label for the x-axis
       y = "Total Sum of Ratings") +  # Label for the y-axis
  theme_minimal()  # Use a minimalistic theme for the plot
  
#6 ======================================================================================
# Calculate the mean rating for each MovieAge in the edx dataset
edx_mean_rating <- edx %>%
  group_by(MovieAge) %>%  # Group data by MovieAge
  summarise(MeanRating = mean(rating, na.rm = TRUE))  # Calculate mean rating, ignoring NA values

# Create a scatter plot to visualize the relationship between MovieAge and MeanRating
ggplot(edx_mean_rating, aes(x = MovieAge, y = MeanRating)) +  # Initialize ggplot with MovieAge on x-axis and MeanRating on y-axis
  geom_point(alpha = 0.5) +  # Add scatter plot points with some transparency for better visualization of data density
  geom_smooth(method = "loess", formula = y ~ x, fill = "gray") +  # Add a LOESS smoothing line to help identify trends
  labs(title = "Movie Age & Average Rating",  # Set the title of the plot
       x = "Movie Age",  # Label for the x-axis
       y = "Mean Rating") +  # Label for the y-axis
  theme_minimal()  # Use a minimalistic theme for a clean and modern look
#7 ======================================================================================
# Summarize and arrange the edx_genres dataset by genre
genres <- edx_genres %>%
  group_by(genres) %>%  # Group data by the 'genres' column
  summarize(count = n()) %>%  # Count the number of rows in each group
  arrange(desc(count))  # Arrange the genres in descending order of count

# Create a bar plot of the number of ratings per genre
genres %>% 
  ggplot(aes(x = reorder(genres, count), y = count)) +  # Initialize ggplot, reordering genres based on count
  geom_bar(stat = 'identity') +  # Create a bar plot with heights equal to 'count' values
  coord_flip() +  # Flip the coordinates to make the plot horizontal
  labs(x = "", y = "Number of Ratings") +  # Set labels for x and y axes (x is left blank)
  geom_text(aes(label = count), hjust = -0.1, size = 3) +  # Add text labels to bars showing the count
  labs(title = "Genres Based on Number of Ratings")  # Set the title of the plot
  
#8 ======================================================================================

# Calculate the mean rating for each genre in the edx_genres dataset
genre_ratings <- edx_genres %>%
  group_by(genres) %>%  # Group data by the 'genres' column
  summarise(MeanRating = mean(rating, na.rm = TRUE)) %>%  # Calculate mean rating for each genre, ignoring NA values
  ungroup() %>%  # Remove the grouping
  arrange(desc(MeanRating))  # Arrange genres in descending order of their mean ratings

# Create a bar plot of the mean ratings for each genre
ggplot(genre_ratings, aes(x = reorder(genres, MeanRating), y = MeanRating)) +  # Initialize ggplot, reordering genres based on mean rating
  geom_bar(stat = "identity") +  # Create a bar plot with heights equal to mean rating values
  coord_flip() +  # Flip the coordinates to make the plot horizontal for better readability
  labs(title = "Mean Rating by Genre",  # Add a title to the plot
       x = "Genres",  # Label for the x-axis
       y = "Mean Rating") +  # Label for the y-axis
  theme_minimal()  # Use a minimalistic theme for a clean and modern look

#9 ======================================================================================
# Grouping, summarizing, and preparing the dataset for plotting
edx_genres %>% 
  group_by(genres) %>%  # Group data by the 'genres' column
  summarize(n = n(),  # Calculate the count of ratings per genre
            avg = mean(rating),  # Calculate the average rating per genre
            se = sd(rating)/sqrt(n())) %>%  # Calculate the standard error of the mean rating
  mutate(genres = reorder(genres, avg)) %>%  # Reorder genres based on the average rating for plotting
  ggplot(aes(x = genres, y = avg, ymin = avg - 2*se, ymax = avg + 2*se)) +  # Create a ggplot object with aesthetics for error bars
  geom_point() +  # Add points for average ratings
  geom_errorbar() +  # Add error bars to represent the standard error
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +  # Adjust the angle of x-axis labels for readability
  labs(title = "Error Bar Plots by Genres",  # Set the title of the plot
       caption = "Source data: edx set")  # Add a caption for data source

# This script creates a plot that visually represents the average rating of each genre, along with error bars indicating the variability of these ratings. It's an effective way to understand the distribution and confidence in the average ratings across different genres.

#10 ======================================================================================

# Prepare the edx_final dataset for training and testing
edx_final <- edx  # Create a copy of the edx dataset

set.seed(1)  # Set a random seed for reproducibility

# Create indices for a test dataset
test_index <- createDataPartition(y = edx_final$rating, times = 1, p = 0.1, list = F)
# 'createDataPartition' function from the caret package is used to split data
# y = edx_final$rating: The splitting is based on the 'rating' variable
# times = 1: Create one set of indices
# p = 0.1: 10% of the data goes into the test set
# list = F: The indices are returned as a vector

# Create the training dataset
trainData <- edx_final[-test_index,]  # Exclude test indices from edx_final to create the training dataset

# Temporary storage of test data
edx_temp <- edx_final[test_index,]  # Create a temporary test dataset using the test indices

# Refine the test dataset
# Ensure that the test dataset only includes movies and users that are also in the training dataset
testData <- edx_temp %>%
  semi_join(trainData, by = "movieId") %>%
  semi_join(trainData, by = "userId")
# 'semi_join' ensures that testData only contains rows with movieId and userId present in trainData

# Identify and remove any observations in the temporary test dataset that are not in the refined test dataset
removed <- anti_join(edx_temp, testData)

# Append these removed observations back to the training dataset
trainData <- rbind(trainData, removed)  # 'rbind' is used to add these observations to the training dataset

# Clean up the environment by removing temporary variables
rm(edx_temp, test_index, removed)  # Remove temporary variables to free up memory and workspace

#11 ======================================================================================

# Calculate the global mean of ratings in the training dataset
edx_train_mu <- mean(trainData$rating) 

# Set a regularization factor to prevent overfitting
lambda2 <- 5

# Calculate bias for each movie ID
b_movieId <- trainData %>% 
  group_by(movieId) %>%
  summarize(b_movieId = sum(rating - edx_train_mu) / (n() + lambda2))
# Group data by movieId
# For each group, calculate the bias as the sum of (rating - global mean) divided by (number of ratings + lambda2)

# Calculate bias for each user ID
b_userId <- trainData %>%
  left_join(b_movieId, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_userId = sum(rating - b_movieId - edx_train_mu) / (n() + lambda2))
# Join with movie bias data
# Group by userId
# For each user, calculate the bias as the sum of (rating - movie bias - global mean) divided by (number of ratings + lambda2)

# Calculate bias for each movie age
b_MovieAge <- trainData %>%
  left_join(b_movieId, by = "movieId") %>% 
  left_join(b_userId, by = "userId") %>%
  group_by(MovieAge) %>%
  summarize(b_MovieAge = sum(rating - b_movieId - b_userId - edx_train_mu) / (n() + lambda2))
# Join with both movie and user biases data
# Group by MovieAge
# For each movie age, calculate the bias as the sum of (rating - movie bias - user bias - global mean) divided by (number of ratings + lambda2)

# These steps are part of a bias calculation typically used in recommendation systems, where biases based on different factors like movie ID, user ID, and movie age are computed to adjust the predictions.

#12 ======================================================================================

# Merging bias terms with the training data
model_data <- trainData %>%
  left_join(b_movieId, by = "movieId") %>%  # Join training data with movie biases using 'movieId'
  left_join(b_userId, by = "userId") %>%  # Join the result with user biases using 'userId'
  left_join(b_MovieAge, by = "MovieAge")  # Finally, join with movie age biases using 'MovieAge'
# This creates a new dataset 'model_data' that includes the original training data along with the calculated biases

# Merging bias terms with the test data
test_data <- testData %>%
  left_join(b_movieId, by = "movieId") %>%  # Join test data with movie biases using 'movieId'
  left_join(b_userId, by = "userId") %>%  # Join the result with user biases using 'userId'
  left_join(b_MovieAge, by = "MovieAge")  # Finally, join with movie age biases using 'MovieAge'
# This creates a new dataset 'test_data' that includes the original test data along with the calculated biases

# These steps are crucial for preparing the data for modeling, as they ensure that both the training and test datasets have the necessary features (biases in this case) required for the model to make accurate predictions.

#13 ======================================================================================

# Fit a linear regression model to the training data
lm_model <- lm(rating ~ b_movieId + b_userId + b_MovieAge, data = model_data)
# 'lm' function is used to fit a linear model
# The model predicts 'rating' based on the biases for movie ID, user ID, and movie age
# 'model_data' is the training dataset that includes these bias terms

# Make predictions on the test data using the fitted model
lm_predictions <- predict(lm_model, newdata = test_data)
# 'predict' function is used to make predictions with the linear model
# 'test_data' is the test dataset, which also includes the bias terms

# Calculate the Root Mean Square Error (RMSE) for the model's predictions
lm_rmse <- sqrt(mean((lm_predictions - testData$rating)^2))
# RMSE is a common measure of the accuracy of predicted numerical values
# It's calculated here as the square root of the mean squared difference between the predicted and actual ratings

# Output the RMSE value
lm_rmse
# This value quantifies the average prediction error in the same units as the ratings (lower values are better)

# This script assesses the performance of a linear regression model built to predict movie ratings. The RMSE provides a measure of the model's prediction error, aiding in evaluating the model's effectiveness.

#14 ======================================================================================

# Prepare the final holdout test data
final_holdout_test_final <- process_movie_data(final_holdout_test)
# The function 'process_movie_data' is applied to the final holdout test dataset
# This function likely performs operations like formatting timestamps and extracting relevant features

# Calculate the overall average rating from the edx_final dataset
edx_final_mu <- mean(edx_final$rating)

# Calculate bias for each movie ID in the edx_final dataset
b_movieId <- edx_final %>% 
  group_by(movieId) %>%
  summarize(b_movieId = sum(rating - edx_final_mu) / (n() + lambda2))
# Group by movieId and calculate the bias for each movie

# Calculate bias for each user ID in the edx_final dataset
b_userId <- edx_final %>% 
  left_join(b_movieId, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_userId = sum(rating - b_movieId - edx_final_mu) / (n() + lambda2))
# Join with movie biases, group by userId, and calculate the user bias

# Calculate bias for each Movie Age in the edx_final dataset
b_MovieAge <- edx_final %>% 
  left_join(b_movieId, by = "movieId") %>% 
  left_join(b_userId, by = "userId") %>%
  group_by(MovieAge) %>%
  summarize(b_MovieAge = sum(rating - b_movieId - b_userId - edx_final_mu) / (n() + lambda2))
# Join with movie and user biases, group by MovieAge, and calculate the Movie Age bias

# Merge the biases into the final holdout test dataset
test_data <- final_holdout_test_final %>%
  left_join(b_movieId, by = "movieId") %>%
  left_join(b_userId, by = "userId") %>%
  left_join(b_MovieAge, by = "MovieAge")

# Make predictions on the final holdout test dataset using the linear model
lm_predictions_validation <- predict(lm_model, newdata = test_data)
# 'predict' function is used with the previously trained linear model 'lm_model'

# Calculate the RMSE for the model's predictions on the final holdout test dataset
lm_rmse_validation <- sqrt(mean((lm_predictions_validation - final_holdout_test_final$rating)^2))
# RMSE is calculated as the square root of the average of the squared differences between predictions and actual ratings

# Output the RMSE value for validation
lm_rmse_validation
# This RMSE value is an indicator of the model's performance on the unseen final holdout test data

# This script is key for evaluating the effectiveness of the linear model on new, unseen data, providing a crucial check on the model's generalizability and predictive accuracy.
