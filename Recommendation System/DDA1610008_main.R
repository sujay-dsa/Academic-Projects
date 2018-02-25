###########################################################
#     Checkpoint 1: (Data Understanding & Data Preparation)
###########################################################
library("arules")
library("arulesViz")
#read data set
global_mart <- read.csv("Global Superstore.csv")


#validate all order dates
d <- try( as.Date( global_mart$Order.Date, format= "%d-%m-%Y" ) )
if( class( d ) == "try-error" || is.na( d ) ) print( "That wasn't correct!" )

View(global_mart)
# all dates in correct format




# We will need to use only order ids and sub categories
transaction_data <- global_mart[,c("Order.ID","Sub.Category")]
transaction_data$Order.ID <- as.factor(transaction_data$Order.ID)
transaction_data$Sub.Category <- as.factor(transaction_data$Sub.Category)
summary(transaction_data)

transaction_data <- as.data.frame(transaction_data)
write.csv(transaction_data, "transaction_data.csv")

transaction_data <- read.transactions(file = "transaction_data.csv",sep = ",",format = "single",
                                      rm.duplicates = T, cols = c("Order.ID","Sub.Category"))

inspect(transaction_data)

itemFrequencyPlot(transaction_data,topN=20,type="absolute")
summary(transaction_data)

# let's optimize parameters to get itemsets of minimum length 2 and support of 0.1, confidence 0.9
rules <- apriori(transaction_data, parameter = list(support = 0.1, confidence = 0.9,
                                                    minlen=2))
# 0 rules created. Let's lower the support and confidence and check
rules <- apriori(transaction_data, parameter = list(support = 0.01, confidence = 0.6,
                                                    minlen=2))

# still 0 rules. Let's lower both support and confidence
rules <- apriori(transaction_data, parameter = list(support = 0.0005, confidence = 0.5,
                                                    minlen=2))

# We get 22 rules. let's check it

inspect(rules)


top10<-head(rules, n=10, by= "confidence")
inspect(top10)

# from the result, we can see that storage, binders are the most 
# popular consequent. Hence, we will analyze the most freqeuent 
# itemsets for these for the convenience of consumers

# For Binders
#----------------------
rulesBinders <- subset(rules, subset = rhs %in% "Binders" & lift > 2.5)
inspect(rulesBinders)
plot(head(rulesBinders, n=5, by = "confidence"),method="graph",interactive=TRUE,shading = "lift")

# From the plot it is evident that binders are most likely to be purchased along with 
# the following
# 1. Paper, Storage, Appliances
unique(global_mart[,c("Category","Sub.Category")])
#    This itemset also seems intuitive, as they're all items one would need in an office.
#       support      = 0.001238266
#       confidence   = 0.6078431
#       lift         = 2.822209
# 2. Furnishing, storage and appliances
#       support      = 0.0011184342
#       confidence   = 0.5957447  
#       lift         = 2.766036




# Storage
rulesStorage <- subset(rules, subset = rhs %in% "Storage" & lift > 2.5)
inspect(rulesStorage)
plot(head(rulesStorage, n=5, by = "confidence"),method="graph",interactive=TRUE,shading = "lift")


# From the plot it is evident that storage is most likely to be purchased along with 
# following itemsets
unique(global_mart[,c("Category","Sub.Category")])
# 1. Furnishings, Binders, Art, Appliances
#       This seems to be a likely itemset as these are items required to set up an
#       office.
#       support      = 0.0005592171
#       confidence   = 0.6086957
#       lift         = 3.360983

# Based on the results, we can say that at least 4 items need to be placed together
# namely: Storage, Binders  Appliances and Furnishings. 



# Based on our previous results, we would like to confirm if we can place furnishings or appliances
# together as these are low selling items. If we can confirm this, we would place it along with
# storage, binders

rules2 <- apriori(transaction_data, parameter = list(support = 0.0005, confidence = 0.5,
                                                     minlen=2, maxlen=4))
inspect(rules2)
top10<-head(rules2, n=10, by= "confidence")
inspect(top10)
rulesFurnishing <- subset(rules2, subset = lhs %in% "Furnishings" & lift > 2.5)
plot(head(rulesFurnishing, n=5, by = "confidence"),method="graph",interactive=TRUE,shading = "lift")

rulesAppliances <- subset(rules2, subset = lhs %in% "Appliances" & lift > 2.5)
plot(head(rulesAppliances, n=5, by = "confidence"),method="graph",interactive=TRUE,shading = "lift")
# From both the plots above it does infact appear that furnishings and appliances usually go together
# along with binders or storage resulting in the purchase of a popular item

# Binders are likely to be picked up 60% of the time if placed along with Storage, Appliances and furnishing
# These are all items related to office supplies and are very good choices to be placed close to
# each other for customer convenience.

# Storage is also likely to be picked up 60% of the time given it is placed close to 
# Furnishings, Binders, Art, Appliances. 

# Effectively, it would be wise to place Furnishings, Binders, Art, Appliances, storage and Paper
# together as these are most likely the items which customers usually buy together. It saves 
# effort and time for the customer to allow them to shop for other products.


