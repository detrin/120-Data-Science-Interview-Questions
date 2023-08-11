## Data Analysis (27 questions)

#### 1. (Given a Dataset) Analyze this dataset and tell me what you can learn from it.
Answer depends on the dataset.
#### 2. What is R2? What are some other metrics that could be better than R2 and why?
  R-squared is a statistical measure that indicates how much of the variation of a dependent variable is explained by an independent variable in a regression model.
#### 3. What is the curse of dimensionality?
  - High dimensionality makes clustering hard, because having lots of dimensions means that everything is "far away" from each other.
  - To cover a fraction of the volume of the data we need to capture a very wide range for each variable as the number of variables increases
  - All samples are close to the edge of the sample. And this is a bad news because prediction is much more difficult near the edges of the training sample.
  - The sampling density decreases exponentially as p increases and hence the data becomes much more sparse without significantly more data. 
#### 4. Is more data always better?
  - Statistically,
    - It depends on the quality of your data, for example, if your data is biased, just getting more data won’t help.
    - It depends on your model. If your model suffers from high bias, getting more data won’t improve your test results beyond a point. You’d need to add more features, etc.
    - In general case we can however say that more data is not going to make the model worse.
  - Practically,
    - Also there’s a tradeoff between having more data and the additional storage, computational power, memory it requires. Hence, always think about the cost of having more data.
#### 5. What are advantages of plotting your data before per- forming analysis?
  - Data sets have errors. Last time I have seen in dataset date with year 2205. 
  - When considering numerical features data can have interesting distributions. Seeing the distribution can help you to think about the nature of predictor. Last time I have spotted that a target has power law distribution. That also makes clear that there is need for target transformation in order to get reasonable error distribution.
#### 6. How can you make sure that you don’t analyze something that ends up meaningless?
Depends how did you get to the data. 
  - When you get the data from database and there is no idea of what those column names mean, someon has to analyze the data and give you some idea of what is in the data. You may find out that some columns are not useful for your analysis. Some may not, but you ahve to analyse them in order to decide.
  - When you get the data from someone or also from databse it is always good to find ppl responsible for those data and ask them what is the meaning of the data. This will help you to make informed decision about the data before any analysis.
#### 7. What is the role of trial and error in data analysis? What is the the role of making a hypothesis before diving in?
I have seen countless times ppl making asumptions about the data. It is always good to direct your energy into some part of the data that you think is important. Especially when human resources are limited and the time is precious. However, I think it is scientific not to be biased in any way and and use only data for making decisions. Making hypothesis before looking at data will help us to stay unbiased. This is one of the worst things that can happen to data analysis or to data scientist. Being biased about things that can have real impact on the business.
#### 8. How can you determine which features are the most important in your model?
Depends on the model and even within model there is usually no definit answer. 
  - In regression models I would look at p-valuesof coefficients. When the task is not to order features by importance, but just select the most important by agreed threshold, I would use Stepwise feature selection or higher order selection used by including interactions.
  - In boosting models I would use shapley values and order the features by importance. When selecting just subset, I would recursively remove N features with lowest importance.
#### 9. How do you deal with some of your predictors being missing?
THis again depends only on the origin of missing values.
  - When the data are missing, but they should not be missing and in production they won't be missing. I would remove the rows with missing values.
  - When the data are missing and they will be missing in production, I would use some of the following techniques:
  - When it is error and there is no meaning by the missing value. I would build another predictive model to predict the missing values - This could be a whole project in itself, so simple techniques are usually used here.
  - When it is not error and the missing value doens't have any special meaning. I would use Weight of Evidence in classification and mean target encoding in regression.
#### 10. You have several variables that are positively correlated with your response, and you think combining all of the variables could give you a good prediction of your response. However, you see that in the multiple linear regression, one of the weights on the predictors is negative. What could be the issue?
Multicollinearity refers to a situation in which two or more explanatory variables in a [multiple regression](https://en.wikipedia.org/wiki/Multiple_regression "Multiple regression") model are highly linearly related. 
  - Leave the model as is, despite multicollinearity. The presence of multicollinearity doesn't affect the efficiency of extrapolating the fitted model to new data provided that the predictor variables follow the same pattern of multicollinearity in the new data as in the data on which the regression model is based.
  - principal component regression
#### 11. Let’s say you’re given an unfeasible amount of predictors in a predictive modeling task. What are some ways to make the prediction more feasible?
  - PCA
  - When we need to use feature selection, we can use one of the following techniques:
    - Univariate Feature Selection where a statistical test is applied to each feature individually. You retain only the best features according to the test outcome scores
    - "Recursive Feature Elimination":  
      - First, train a model with all the feature and evaluate its performance on held out data.
      - Then drop let say the 10% weakest features (e.g. the feature with least absolute coefficients in a linear model) and retrain on the remaining features.
      - Iterate until you observe a sharp drop in the predictive accuracy of the model.
    - Hierarchical variable clustering.
#### 12. Now you have a feasible amount of predictors, but you’re fairly sure that you don’t need all of them. How would you perform feature selection on the dataset?
  - Univariate Feature Selection where a statistical test is applied to each feature individually. You retain only the best features according to the test outcome scores
  - "Recursive Feature Elimination":  
    - First, train a model with all the feature and evaluate its performance on held out data.
    - Then drop let say the 10% weakest features (e.g. the feature with least absolute coefficients in a linear model) and retrain on the remaining features.
    - Iterate until you observe a sharp drop in the predictive accuracy of the model.
  - Hierarchical variable clustering.
#### 13. Your linear regression didn’t run and communicates that there are an infinite number of best estimates for the regression coefficients. What could be wrong?
If some of the explanatory variables are perfectly correlated (positively or negatively) then the coefficients would not be unique. 
#### 14. You run your regression on different subsets of your data, and find that in each subset, the beta value for a certain variable varies wildly. What could be the issue here?
The dataset might be heterogeneous. In which case, it is recommended to cluster datasets into different subsets wisely, and then draw different models for different subsets. Or, use models like non parametric models (trees) which can deal with heterogeneity quite nicely.
#### 15. What is the main idea behind ensemble learning? If I had many different models that predicted the same response variable, what might I want to do to incorporate all of the models? Would you expect this to perform better than an individual model or worse?
  - The assumption is that a group of weak learners can be combined to form a strong learner.
  - Hence the combined model is expected to perform better than an individual model.
  - Assumptions:
    - average out biases
    - reduce variance
  - Bagging works because some underlying learning algorithms are unstable: slightly different inputs leads to very different outputs. If you can take advantage of this instability by running multiple instances, it can be shown that the reduced instability leads to lower error. If you want to understand why, the original bagging paper( [http://www.springerlink.com/cont...](http://www.springerlink.com/content/l4780124w2874025/)) has a section called "why bagging works"
  - Boosting works because of the focus on better defining the "decision edge". By reweighting examples near the margin (the positive and negative examples) you get a reduced error (see http://citeseerx.ist.psu.edu/vie...)
  - Use the outputs of your models as inputs to a meta-model.   

For example, if you're doing binary classification, you can use all the probability outputs of your individual models as inputs to a final logistic regression (or any model, really) that can combine the probability estimates.  

One very important point is to make sure that the output of your models are out-of-sample predictions. This means that the predicted value for any row in your dataframe should NOT depend on the actual value for that row.
#### 16. Given that you have wi  data in your o ce, how would you determine which rooms and areas are underutilized and overutilized?
  - If the data is more used in one room, then that one is over utilized! Maybe account for the room capacity and normalize the data.
#### 17. How could you use GPS data from a car to determine the quality of a driver?
We could use unsupervised learning to compare one driver to others. 
  - I would first think of sensible features that will enable us to run model on tbaular data.
  - Then I would use various clusetring methods to see whether there are any obvious clusers in the dataset or not. Perhaps cluster could be labeled as how much quality driver is.
  - I would also look how much we are able to reduce the dimensionality. If the dimensionality of data could be significantly reduced I would try to draw boundaries on this reduced dimensionality space while inspecting single cases of drivers. 

#### 18. Given accelerometer, altitude, and fuel usage data from a car, how would you determine the optimum acceleration pattern to drive over hills?
Depends optimal in what terms. If it is optimal in the amount of fuel used and considering I have many histories of the same car. I would segment altitude into let say 100 bins with same size. Then I would calculate for each history power needed for car to pass that bin. Power could be calculated as acceleration times delta T and integrated over each bin. Then I would use simple regression to see what are the coefficients in respect to used fuel and look for minimum value, but still within the range of data. After this analysis I would convert acceleration into velocity over altitude and see if there are clean patterns that could be correlated to coefficients. Alternatively, we could make the bins finer and finer for altitude and see which parts of the hill are problematic and where car needs to accelerate more in order to preserve constant momentum.

#### 19. Given position data of NBA players in a season’s games, how would you evaluate a basketball player’s defensive ability?
I would classify positions into defensive and not-defensive. Every person has his own performance in defensive and not-defensive position. I would assume that those are two constants that stays same during the whole season. Each match is result of those capabilities of players. We would need to constrain the capabilities let's say to range [0, 1]. Defensife weights have negative effect on second team score and offensive weight have positive effect on the teams score. Hence it can be reduced to just two total weights that will be normalized and used to calculate the score with sampling. The whole task definition is Beta distribution. In the first approximation we can assume that the weight ration should resemble the score ratio. Hence we can use player's weigths as variables that should be optimized. Their restriction as boudnary conditions. The difference between ratios and weights and their sum is cost function. Now we need to just optimize those weights in order to get player's defensive capabilities. We could also directly use dirichlet sitribution to maximize the likelihood.
#### 20. How would you quantify the influence of a Twitter user?
  - I would obtain data on post during past 12 months, such as number of likes, amount of shares amount of views and amoutn of followers during each month.
  - Then I would calculate average of those per post during each of 12 months. 
  - Try to cluster the users based on those features and see if it could be reduced with PCA. I would also try to make them relative to the beginning 12 months ago and keep that factor outside. 
  - I would aim to extract the grow factor and define it in simple terms based on those features and current reach. Both should be taken into account when trying to work with influence or predict influence of user on next month let's say.
  
#### 21. Given location data of golf balls in games, how would construct a model that can advise golfers where to aim?
Each position (3 dim) has also direction where the player was shooting if we don't consider the force with which the ball was hit (1 dim). Since we have only one variable that we want to predict I would use kriging. This is also assuming the shots are not dependend on each other in respect to the final position. 
#### 22. You have 100 mathletes and 100 math problems. Each mathlete gets to choose 10 problems to solve. Given data on who got what problem correct, how would you rank the problems in terms of difficulty?
  - One way you could do this is by storing a "skill level" for each user and a "difficulty level" for each problem.  We assume that the probability that a user solves a problem only depends on the skill of the user and the difficulty of the problem.*  Then we maximize the likelihood of the data to find the hidden skill and difficulty levels.
  - The Rasch model for dichotomous data takes the form:  
{\displaystyle \Pr\\{X_{ni}=1\\}={\frac {\exp({\beta _{n}}-{\delta _{i}})}{1+\exp({\beta _{n}}-{\delta _{i}})}},}  
where  is the ability of person  and  is the difficulty of item}.
#### 23. You have 5000 people that rank 10 sushis in terms of saltiness. How would you aggregate this data to estimate the true saltiness rank in each sushi?
  - Using median for every sushi is not bad start.
  - However we ar emore interested in ranking the sushi by saltiness. The better idea then would be to weight all the votes of person by the total number of votes of that person. Then we can calculate the median of weighted votes for each sushi and rank them by that median.
#### 24. Given data on congressional bills and which congressional representatives co-sponsored the bills, how would you determine which other representatives are most similar to yours in voting behavior? How would you evaluate who is the most liberal? Most republican? Most bipartisan?
  - collaborative filtering. you have your votes and we can calculate the similarity for each representatives and select the most similar representative
  - for liberal and republican parties, find the mean vector and find the representative closest to the center point
#### 25. How would you come up with an algorithm to detect plagiarism in online content?
  - bag of words on words that are less often used in regular text
  - use relative frequency of words in the text and they relative position in text to create signiture of words
  - use cosine similarity to compare the signitures
#### 26. You have data on all purchases of customers at a grocery store. Describe to me how you would program an algorithm that would cluster the customers into groups. How would you determine the appropriate number of clusters to include?
  - KMeans
  - choose a small value of k that still has a low SSE (elbow method)
  - <https://bl.ocks.org/rpgove/0060ff3b656618e9136b>
#### 27. Let's say you're building the recommended music engine at Spotify to recommend people music based on past listening history. How would you approach this problem?
  - [collaborative filtering](https://en.wikipedia.org/wiki/Collaborative_filtering)
