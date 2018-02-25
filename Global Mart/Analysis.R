#####################################################################################
#
#                         CHECKPOINT 1
#
#####################################################################################
global_mart <- read.csv(file.choose(), stringsAsFactors = FALSE)
str(global_mart)

#check for missing or NA values
sapply(global_mart, function(x) sum(is.na(x)))
# Eliminate postal code data, row id, customer name and product name 
# since they are irrelevant to model building
global_mart_pruned <- global_mart[, c(-1, -7, -12, -18)]

#change the column type appropriately
str(global_mart_pruned)
sapply(global_mart_pruned, function(x) sum(is.na(x)))

global_mart_pruned$Order.ID <- as.factor(global_mart_pruned$Order.ID)
str(global_mart_pruned$Order.ID)
duplicated_orders <- which(duplicated(global_mart_pruned$Order.ID))
unique_orders <- unique(global_mart_pruned)

#---------------------------------------------------------------
# CHECKPOINT 1A - create combinations of markets and segments
#---------------------------------------------------------------
global_mart_pruned$market_segment <- paste(global_mart_pruned$Market, global_mart_pruned$Segment)
global_mart_pruned$market_segment <- as.character(global_mart_pruned$market_segment)
summary(as.factor(global_mart_pruned$market_segment))

#---------------------------------------------------------------
# CHECKPOINT 1B - aggregate transactional data at monthly level for each 
#                 of the 21 subsets
#---------------------------------------------------------------
# 
# first step would be to extract the month
global_mart_pruned$month_ordered <- months(as.Date(global_mart_pruned$Order.Date))


global_mart_pruned$month_ordered <- as.factor(global_mart_pruned$month_ordered)
summary(global_mart_pruned$month_ordered)

#check point 2 - aggregate transaction data monthly for Profit for each of the 21 segmentsss
global_mart_pruned$OrderDate <- as.Date(global_mart_pruned$Order.Date, format = "%d-%m-%Y")
mo <- strftime(global_mart_pruned$OrderDate, "%m")
yr <- strftime(global_mart_pruned$OrderDate, "%Y")

global_mart_pruned$Month_And_Year <- (paste(yr ,mo,  sep = "-"))
head(global_mart_pruned$Month_And_Year)
global_profit_by_month <-
  aggregate(cbind(Profit,Sales) ~ market_segment +  Month_And_Year , data = global_mart_pruned, sum)

############################################################
#Plot to examine trends
############################################################
library(ggplot2)

ggplot(
  global_profit_by_month[order(global_profit_by_month$Profit, decreasing = TRUE), ],
  aes(
    Month_And_Year,
    Profit,
    colour = market_segment,
    group = market_segment,
    label = market_segment
  )
) + geom_line()


#---------------------------------------------------------------
# CHECKPOINT 1C - calculate coefficient of variation
#---------------------------------------------------------------

#calculate coefficient of variation for the most profitable and consistently profitable market segment
# should aggregate with month + year

#Profit
top_profitable_segments <-
  as.data.frame(as.list(
    aggregate(
      cbind(Profit,Sales) ~ market_segment,
      data = global_profit_by_month,
      FUN = function(x)
        c(cv = sd(x) / mean(x), tot = sum(x))
    )
  ))
top_profitable_segments[order(top_profitable_segments$Profit.tot, decreasing = TRUE), ]
top_profitable_segments[order(top_profitable_segments$Profit.cv, decreasing = FALSE), ]
#based on CV, US consumer, APAC Corporate, EU Corporate, LATAM Consumer, APAC Consumer can be chosen
top_profitable_segments <- top_profitable_segments[c(13, 4, 16, 5, 14), ]
top_markets_by_profit <-
  global_mart_pruned[global_mart_pruned$market_segment %in% top_profitable_segments$market_segment, ]

summary(top_markets_by_profit)

#---------------------------------------------------------------
# CHECKPOINT 1D - separate out last 6 months data
#---------------------------------------------------------------
#split the data for each of the metrics for the last 6 months, i.e. from July to december

top_markets_by_profit$Order.Date <- as.Date(top_markets_by_profit$Order.Date, format = "%d-%m-%Y")
summary(top_markets_by_profit$Order.Date)

# last order was on december 31 2014. So extract last 6 months into a test frame and remaining into a training frame
top_markets_by_profit_train <- 
  top_markets_by_profit[top_markets_by_profit$Order.Date < as.Date("2014-06-30"), ]
str(top_markets_by_profit)

#aggregate by month
aggregated_sales <-  
  aggregate(cbind(Sales,Quantity) ~ Month_And_Year + market_segment, 
            top_markets_by_profit_train, FUN = sum)
head(aggregated_sales)

top_markets_by_profit_test <-
  top_markets_by_profit[top_markets_by_profit$Order.Date > as.Date("2014-06-30"), ]

aggregated_sales_test <- aggregate(cbind(Sales,Quantity) ~ Month_And_Year + market_segment, top_markets_by_profit_test, FUN = sum)
head(aggregated_sales_test)

cols <- c("red", "blue")
labels <- c("Raw", "Smoothed")
xlab1 <- c("Months from Jan 2011")


summary(as.factor(aggregated_sales$market_segment))
timeser <- ts(aggregated_sales[aggregated_sales$market_segment=="EU Consumer",]$Sales ,start=c(2011,1), end=c(2014,6), frequency=12)
ts_APAC_Cons <- ts(aggregated_sales[aggregated_sales$market_segment=="APAC Consumer",]$Sales, start=c(2011,1), end = c(2014,6), frequency = 12)
ts_APAC_Corp <- ts(aggregated_sales[aggregated_sales$market_segment=="APAC Corporate",]$Sales, start=c(2011,1), end = c(2014,6), frequency = 12)
ts_EU_corp <- ts(aggregated_sales[aggregated_sales$market_segment=="EU Corporate",]$Sales, start=c(2011,1), end = c(2014,6), frequency = 12)
ts_LATAM_cons <- ts(aggregated_sales[aggregated_sales$market_segment=="LATAM Consumer",]$Sales, start=c(2011,1), end = c(2014,6), frequency = 12)

plot(timeser)
timeser_Qty <- ts(aggregated_sales[aggregated_sales$market_segment=="EU Consumer",]$Quantity, start=c(2011,1), end = c(2014,6), frequency = 12)
ts_APAC_Cons_Qty <- ts(aggregated_sales[aggregated_sales$market_segment=="APAC Consumer",]$Quantity, start=c(2011,1), end = c(2014,6), frequency = 12)
ts_APAC_Corp_Qty <- ts(aggregated_sales[aggregated_sales$market_segment=="APAC Corporate",]$Quantity, start=c(2011,1), end = c(2014,6), frequency = 12)
ts_EU_corp_Qty <- ts(aggregated_sales[aggregated_sales$market_segment=="EU Corporate",]$Quantity, start=c(2011,1), end = c(2014,6), frequency = 12)
ts_LATAM_cons_Qty <- ts(aggregated_sales[aggregated_sales$market_segment=="LATAM Consumer",]$Quantity, start=c(2011,1), end = c(2014,6), frequency = 12)

timeser_test <- ts(aggregated_sales_test[aggregated_sales_test$market_segment=="EU Consumer",]$Sales, start=c(2014,7), end = c(2014,12), frequency = 12)
ts_APAC_Cons_test <- ts(aggregated_sales_test[aggregated_sales_test$market_segment=="APAC Consumer",]$Sales, start=c(2014,7), end = c(2014,12), frequency = 12)
ts_APAC_Corp_test <- ts(aggregated_sales_test[aggregated_sales_test$market_segment=="APAC Corporate",]$Sales, start=c(2014,7), end = c(2014,12), frequency = 12)
ts_EU_corp_test <- ts(aggregated_sales_test[aggregated_sales_test$market_segment=="EU Corporate",]$Sales, start=c(2014,7), end = c(2014,12), frequency = 12)
ts_LATAM_cons_test <- ts(aggregated_sales_test[aggregated_sales_test$market_segment=="LATAM Consumer",]$Sales, start=c(2014,7), end = c(2014,12), frequency = 12)

timeser_test_qty <- ts(aggregated_sales_test[aggregated_sales_test$market_segment=="EU Consumer",]$Quantity, start=c(2014,7), end = c(2014,12), frequency = 12)
ts_APAC_Cons_test_qty <- ts(aggregated_sales_test[aggregated_sales_test$market_segment=="APAC Consumer",]$Quantity, start=c(2014,7), end = c(2014,12), frequency = 12)
ts_APAC_Corp_test_qty <- ts(aggregated_sales_test[aggregated_sales_test$market_segment=="APAC Corporate",]$Quantity, start=c(2014,7), end = c(2014,12), frequency = 12)
ts_EU_corp_test_qty <- ts(aggregated_sales_test[aggregated_sales_test$market_segment=="EU Corporate",]$Quantity, start=c(2014,7), end = c(2014,12), frequency = 12)
ts_LATAM_cons_test_qty <- ts(aggregated_sales_test[aggregated_sales_test$market_segment=="LATAM Consumer",]$Quantity, start=c(2014,7), end = c(2014,12), frequency = 12)

segment_titles <- c("EU Consumer","APAC Consumer", "APAC Corporate","EU Corporate","LATAM Consumer")

#---------------------------------------------------------------
#
# CHECKPOINT 2 - Plot TIME SERIES / SMOOTHEN TIME SERIES / 
#               CHECK RESIDUALS FOR WHITE NOISE/
#               FIND OPTIMAL P,D,Q 
#
#---------------------------------------------------------------
# Function to plot the smooothed series for any noisy time series
plot_smoothed_series <- function(width, timeser,title,ylab1) {
  smoothedseries <- filter(
    timeser,
    filter = rep(1 / (2 * width + 1), (2 * width + 1)),
    method = "convolution",
    sides = 2
  )
  
  # fill the first value
  w <- 1
  diff <- smoothedseries[w + 2] - smoothedseries[w + 1]
  for (i in seq(w, 1, -1)) {
    smoothedseries[i] <- smoothedseries[i + 1] - diff
  }
  
  # fill the last value
  n <- length(timeser)
  timevalues_smoothed_series <- c(1:42)
  diff <- smoothedseries[n - w] - smoothedseries[n - w - 1]
  for (i in seq(n - w + 1, n)) {
    smoothedseries[i] <- smoothedseries[i - 1] + diff
  }
  
  plot(timeser,
       main = title,
       xlab = xlab1,
       ylab = ylab1)
  lines(smoothedseries, col = "blue", lwd = 2)
}

#---------------------------------------------
#                     EU Consumer
#--------------------------------------------


library(stats)
library(forecast)

plot(decompose(timeser))
#From the decompose plot, we see that the model is additive
lmfit <- tslm(timeser ~ trend + season, data = timeser)
predicted_timeser <- ts(predict(lmfit), start=c(2011,1),end = c(2014,6), frequency=12)
timeser
predicted_timeser
residue <- predicted_timeser - timeser
plot_smoothed_series(2,timeser, "EU Consumer Sales From Jan 2011 to Jun 2014","Sales")
lines(predicted_timeser, col="red")


# check for noise
qqnorm(residue) # 45 degree line implies noise
acf(residue)
pacf(residue)
auto.arima(residue) # 0,0,0 model which is essentially noise
hist(residue)

# auto arima modelling
autoarima <- auto.arima(timeser)
autoarima
arima_model_EU_Cons_Sales <- arima(x = timeser,order = c(2,1,0))

# it is an ar2 process p=2,d=1,q=0
acf(diff(timeser,differences = 1)) # fall off to zero immediately
pacf(diff(timeser, differences = 1)) # fall off to zero at lag 2

############
# For Qty
############
plot_smoothed_series(2,timeser_Qty,"EU Consumer Quantity From Jan 2011 to Jun 2014","Quantity")
plot(decompose(timeser_Qty))
#From the decompose plot, we see that the model is additive
lmfit <-tslm(timeser_Qty ~ trend + season, data = timeser_Qty)
predicted_timeser <- ts(predict(lmfit), start=c(2011,1),end = c(2014,6), frequency=12)
predicted_timeser
plot_smoothed_series(2,timeser_Qty,"EU Consumer Quantity From Jan 2011 to Jun 2014","Quantity")
lines(predicted_timeser, col="red")

# check for noise
residue <- predicted_timeser - timeser_Qty
qqnorm(residue) # 45 degree line implies noise
acf(residue)
pacf(residue)
auto.arima(residue) # 0,0,0 model which is essentially noise
hist(residue)

#auto arima modelling
autoarima <- auto.arima(timeser_Qty)
arima_model_EU_Cons_Qty <- arima(x = timeser_Qty,order = c(2,1,0))

# it is an ar2 process p=2,d=1,q=0
acf(diff(timeser_Qty,differences = 1)) 
pacf(diff(timeser_Qty, differences = 1)) 

# ---------------------------------------------
#                     APAC Consumer
# --------------------------------------------
plot_smoothed_series(2,ts_APAC_Cons,"APAC Consumer Sales From Jan 2011 to Jun 2014")
plot(decompose(ts_APAC_Cons))
#From the decompose plot, we see that the model is additive
lmfit <-tslm(ts_APAC_Cons ~ trend + season, data = ts_APAC_Cons)
predicted_timeser <- ts(predict(lmfit), start=c(2011,1),end = c(2014,6), frequency=12)
predicted_timeser
plot_smoothed_series(2,ts_APAC_Cons,"APAC Consumer Sales From Jan 2011 to Jun 2014","Sales")
lines(predicted_timeser, col="red")

# check for noise
residue <- predicted_timeser - ts_APAC_Cons
qqnorm(residue) # 45 degree line implies noise
acf(residue)
pacf(residue)
auto.arima(residue) # 0,0,0 model which is essentially noise
hist(residue)

#auto arima modelling
autoarima <- auto.arima(ts_APAC_Cons)
arima_model_APAC_Cons_Sales <- arima(x = ts_APAC_Cons,order = c(0,1,1))

# it is an ma1 process p=0,d=1,q=1
acf(diff(ts_APAC_Cons,differences = 1)) 
pacf(diff(ts_APAC_Cons, differences = 1))

# Quantity
plot_smoothed_series(2,ts_APAC_Cons_Qty,"APAC Consumer Qty From Jan 2011 to Jun 2014","Quantity")
plot(decompose(ts_APAC_Cons_Qty))
#From the decompose plot, we see that the model is additive
lmfit <-tslm(ts_APAC_Cons_Qty ~ trend + season, data = ts_APAC_Cons_Qty)
predicted_timeser <- ts(predict(lmfit), start=c(2011,1),end = c(2014,6), frequency=12)
predicted_timeser
plot_smoothed_series(2,ts_APAC_Cons_Qty,"APAC Consumer Qty From Jan 2011 to Jun 2014","Quantity")
lines(predicted_timeser, col="red",lwd=2)

# check for noise
residue <- predicted_timeser - ts_APAC_Cons_Qty
qqnorm(residue) # 45 degree line implies noise
acf(residue)
pacf(residue)
auto.arima(residue) # 0,0,0 model which is essentially noise
hist(residue)

#auto arima modelling
autoarima <- auto.arima(ts_APAC_Cons_Qty)
arima_model_APAC_Cons_Qty <- arima(x = ts_APAC_Cons_Qty,order = c(0,1,0))

# it is an ma1 process p=0,d=1,q=0
acf(diff(ts_APAC_Cons_Qty,differences = 1)) 
pacf(diff(ts_APAC_Cons_Qty, differences = 1))

# ---------------------------------------------
#                     APAC Corporate
# --------------------------------------------
# Sales
plot_smoothed_series(2,ts_APAC_Corp,"APAC Corporate Sales From Jan 2011 to Jun 2014","Sales")
plot(decompose(ts_APAC_Corp))
#From the decompose plot, we see that the model is additive
lmfit <-tslm(ts_APAC_Corp ~ trend + season, data = ts_APAC_Corp)
predicted_timeser <- ts(predict(lmfit), start=c(2011,1),end = c(2014,6), frequency=12)
predicted_timeser
plot_smoothed_series(2,ts_APAC_Corp,"APAC Corporate Sales From Jan 2011 to Jun 2014","Sales")
lines(predicted_timeser, col="red")

# check for noise
residue <- predicted_timeser - ts_APAC_Corp
qqnorm(residue) # 45 degree line implies noise
acf(residue)
pacf(residue)
auto.arima(residue) # 0,0,0 model which is essentially noise
hist(residue)

#auto arima modelling
autoarima <- auto.arima(ts_APAC_Corp)
arima_model_APAC_Corp_Sales<- arima(x = ts_APAC_Corp,order = c(0,1,1))

# it is an ma1 process p=0,d=1,q=1
acf(diff(ts_APAC_Corp,differences = 1)) 
pacf(diff(ts_APAC_Corp, differences = 1))

# Quantity
plot_smoothed_series(2,ts_APAC_Corp_Qty,"APAC Corporate Qty From Jan 2011 to Jun 2014","Quantity")
plot(decompose(ts_APAC_Corp_Qty))
#From the decompose plot, we see that the model is additive
lmfit <-tslm(ts_APAC_Corp_Qty ~ trend + season, data = ts_APAC_Corp_Qty)
predicted_timeser <- ts(predict(lmfit), start=c(2011,1),end = c(2014,6), frequency=12)
predicted_timeser
plot_smoothed_series(2,ts_APAC_Corp_Qty,"APAC Corporate Qty From Jan 2011 to Jun 2014","Quantity")
lines(predicted_timeser, col="red",lwd=2)

# check for noise
residue <- predicted_timeser - ts_APAC_Corp_Qty
qqnorm(residue) # 45 degree line implies noise
acf(residue)
pacf(residue)
auto.arima(residue) # 0,0,0 model which is essentially noise
hist(residue)

#auto arima modelling
autoarima <- auto.arima(ts_APAC_Corp_Qty)
arima_model_APAC_Corp_Qty<- arima(x = ts_APAC_Corp_Qty,order = c(0,1,1))

# it is an ma1 process p=0,d=1,q=1
acf(diff(ts_APAC_Corp_Qty,differences = 1)) 
pacf(diff(ts_APAC_Corp_Qty, differences = 1))

# --------------------------------------------
#                     EU Corporate
# --------------------------------------------
# Sales
plot_smoothed_series(2,ts_EU_corp,"EU Corporate Sales From Jan 2011 to Jun 2014","Sales")
plot(decompose(ts_EU_corp))
#From the decompose plot, we see that the model is additive
lmfit <-tslm(ts_EU_corp ~ trend + season, data = ts_EU_corp)
predicted_timeser <- ts(predict(lmfit), start=c(2011,1),end = c(2014,6), frequency=12)
predicted_timeser
plot_smoothed_series(2,ts_EU_corp,"EU Corporate Sales From Jan 2011 to Jun 2014","Sales")
lines(predicted_timeser, col="red")

# check for noise
residue <- predicted_timeser - ts_EU_corp
qqnorm(residue) # 45 degree line implies noise
acf(residue)
pacf(residue)
auto.arima(residue) # 0,0,1 model with 0 mean. Almost close to noise
hist(residue)

# auto arima modelling
autoarima <- auto.arima(ts_EU_corp)
arima_model_EU_Corp_Sales<- arima(x = ts_EU_corp,order = c(2,1,0))

# it is an ma1 process p=2,d=1,q=0
acf(diff(ts_EU_corp,differences = 1)) 
pacf(diff(ts_EU_corp, differences = 1))

# Quantity
plot_smoothed_series(2,ts_EU_corp_Qty,"EU Corporate Qty From Jan 2011 to Jun 2014","Quantity")
plot(decompose(ts_EU_corp_Qty))
#From the decompose plot, we see that the model is additive
lmfit <-tslm(ts_EU_corp_Qty ~ trend + season, data = ts_EU_corp_Qty)
predicted_timeser <- ts(predict(lmfit), start=c(2011,1),end = c(2014,6), frequency=12)
predicted_timeser
plot_smoothed_series(2,ts_EU_corp_Qty,"EU Corporate Qty From Jan 2011 to Jun 2014","Quantity")
lines(predicted_timeser, col="red",lwd=2)

# check for noise
residue <- predicted_timeser - ts_EU_corp_Qty
qqnorm(residue) # 45 degree line implies noise
acf(residue)
pacf(residue)
auto.arima(residue) # 0,0,1 model with 0 mean. Almost close to noise
hist(residue)

# auto arima modelling
autoarima <- auto.arima(ts_EU_corp_Qty)
arima_model_EU_Corp_Qty<- arima(x = ts_EU_corp_Qty,order = c(2,1,0))

# it is an ma1 process p=2,d=1,q=0
acf(diff(ts_EU_corp_Qty,differences = 1)) 
pacf(diff(ts_EU_corp_Qty, differences = 1))

# ---------------------------------------------
#                     LATAM Consumer
# --------------------------------------------
# Sales
plot_smoothed_series(2, ts_LATAM_cons,"LATM Consumer Sales from Jan 2011 to Jun 2014","Sales")
plot(decompose(ts_LATAM_cons))
#From the decompose plot, we see that the model is additive
lmfit <-tslm(ts_LATAM_cons ~ trend + season, data = ts_LATAM_cons)
predicted_timeser <- ts(predict(lmfit), start=c(2011,1),end = c(2014,6), frequency=12)
predicted_timeser
plot_smoothed_series(2, ts_LATAM_cons,"LATM Consumer Sales from Jan 2011 to Jun 2014","Sales")
lines(predicted_timeser, col="red")

# check for noise
residue <- predicted_timeser - ts_LATAM_cons
qqnorm(residue) # 45 degree line implies noise
acf(residue)
pacf(residue)
auto.arima(residue) # 0,0,0 model with 0 mean. Almost close to noise
hist(residue)

# auto arima modelling
autoarima <- auto.arima(ts_LATAM_cons)
arima_model_LATAM_Cons_Sales<- arima(x = ts_LATAM_cons,order = c(0,1,0))

# it is an ma1 process p=0,d=1,q=0
acf(diff(ts_LATAM_cons,differences = 1)) 
pacf(diff(ts_LATAM_cons, differences = 1))

# Quantity
plot_smoothed_series(2,ts_LATAM_cons_Qty,"LATM Consumer Qty from Jan 2011 to Jun 2014","Quantity")
plot(decompose(ts_LATAM_cons_Qty))
#From the decompose plot, we see that the model is additive
lmfit <-tslm(ts_LATAM_cons_Qty ~ trend + season, data = ts_LATAM_cons_Qty)
predicted_timeser <- ts(predict(lmfit), start=c(2011,1),end = c(2014,6), frequency=12)
predicted_timeser
plot_smoothed_series(2,ts_LATAM_cons_Qty,"LATM Consumer Qty from Jan 2011 to Jun 2014","Quantity")
lines(predicted_timeser, col="red",lwd=2)

# check for noise
residue <- predicted_timeser - ts_LATAM_cons_Qty
qqnorm(residue) # 45 degree line implies noise
acf(residue)
pacf(residue)
auto.arima(residue) # 0,0,0 model with 0 mean. Almost close to noise
hist(residue)

# auto arima modelling
autoarima <- auto.arima(ts_LATAM_cons_Qty)
arima_model_LATAM_Cons_Qty<- arima(x = ts_LATAM_cons_Qty,order = c(0,1,0))

# it is an ma1 process p=0,d=1,q=0
acf(diff(ts_LATAM_cons_Qty,differences = 1)) 
pacf(diff(ts_LATAM_cons_Qty, differences = 1))

#-------------------------------------------------------
#
#     CHECKPOINT 3 : MODEL EVALUATION
#
#-------------------------------------------------------

mape <- function(residue, timeser){
  sum(abs(residue/timeser))/length(timeser)
}

# ---------------------------------------------
#                     EU Consumer
# --------------------------------------------
# Testing Sales
total_time_series <- ts(c(timeser,timeser_test),start=c(2011,1),end = c(2014,12), frequency=12)
plot_smoothed_series(2,total_time_series,"EU Consumer Sales from Jan 2011 to Dec 2014","Sales")
lmfit_EU_Cons <-tslm(total_time_series ~ trend + season, data = total_time_series)
predicted_timeser <- ts(predict(lmfit_EU_Cons), start=c(2011,1),end = c(2014,12), frequency=12)
predicted_timeser
plot_smoothed_series(2,total_time_series,"EU Consumer Sales from Jan 2011 to Dec 2014","Sales")
lines(predicted_timeser, col="red")

# check for noise
residue <- predicted_timeser - total_time_series
# Forecasting using arima
arima_forecast <- forecast.Arima(arima_model_EU_Cons_Sales,h = 6)
arima_residue <- ts(arima_forecast$mean, start = c(2014,7), end=c(2014,12),frequency = 12) - timeser_test

#-----------------------------------------------------
# Mape value for regression model
mape_regression <-  mape(residue[43:48], total_time_series[43:48]) #9.4%
mape_regression
#Mape value for ARIMA
mape_arima <- mape(arima_residue, timeser_test) # 28.9%
mape_arima
#-----------------------------------------------------

# Choosing Regression as the final model
# Predicting from Jan 2015 and Jun 2015
EU_Cons_Sales_pred <- forecast(lmfit_EU_Cons,h=6)
EU_Cons_Sales_pred

#-----------------
# Testing Quantity
#-----------------
total_time_series <- ts(c(timeser_Qty,timeser_test_qty),start=c(2011,1),end = c(2014,12), frequency=12)
plot_smoothed_series(2,total_time_series,"EU Consumer Qty from Jan 2011 to Dec 2014","Quantity")
lmfit_EU_Cons <-tslm(total_time_series ~ trend + season, data = total_time_series)
predicted_timeser <- ts(predict(lmfit_EU_Cons), start=c(2011,1),end = c(2014,12), frequency=12)
predicted_timeser
plot_smoothed_series(2,total_time_series,"EU Consumer Qty from Jan 2011 to Dec 2014","Quantity")
lines(predicted_timeser, col="red",lwd=2)

# check for noise
residue <- predicted_timeser - total_time_series

# Forecasting using arima
arima_forecast <- forecast.Arima(arima_model_EU_Cons_Qty,h = 6)
arima_residue <- ts(arima_forecast$mean, start = c(2014,7), end=c(2014,12),frequency = 12) - timeser_test_qty
#-----------------------------------------------------

# Mape value for regression model
mape_regression <-  mape(residue[43:48], total_time_series[43:48]) # 10.43%
mape_regression
#Mape value for ARIMA
mape_arima <- mape(arima_residue, timeser_test_qty) # 30.21%
mape_arima
#-----------------------------------------------------
# Choosing Regression as the final model
# Predicting from Jan 2015 and Jun 2015
EU_Cons_Qty_pred <- forecast(lmfit_EU_Cons,h=6)
EU_Cons_Qty_pred

# ---------------------------------------------
#                     APAC Consumer
# --------------------------------------------
#Testing for sales
total_time_series <- ts(c(ts_APAC_Cons,ts_APAC_Cons_test),start=c(2011,1),end = c(2014,12), frequency=12)
lmfit <-tslm(total_time_series ~ trend + season, data = total_time_series)
predicted_timeser <- ts(predict(lmfit), start=c(2011,1),end = c(2014,12), frequency=12)
predicted_timeser
plot_smoothed_series(2,total_time_series,"APAC Consumer Sales from Jan 2011 to Dec 2014","Sales")
lines(predicted_timeser, col="red")

# check for noise
residue <- predicted_timeser - total_time_series

# Forecasting using arima
arima_forecast <- forecast.Arima(arima_model_APAC_Cons_Sales,h = 6)
arima_residue <- ts(arima_forecast$mean, start = c(2014,7), end=c(2014,12),frequency = 12) - ts_APAC_Cons_test
#-----------------------------------------------------

# Mape value for regression model
mape_regression <-  mape(residue[43:48], ts_APAC_Cons_test) # 9.88%
mape_regression
#Mape value for ARIMA
mape_arima <- mape(arima_residue, ts_APAC_Cons_test) # 27.7%
mape_arima
#-----------------------------------------------------
# Choosing Regression as the final model
# Predicting from Jan 2015 and Jun 2015
APAC_Cons_Sales_pred <- forecast(lmfit,h=6)
APAC_Cons_Sales_pred

# Testing for Quantity
total_time_series <- ts(c(ts_APAC_Cons_Qty,ts_APAC_Cons_test_qty),start=c(2011,1),end = c(2014,12), frequency=12)
lmfit <-tslm(total_time_series ~ trend + season, data = total_time_series)
predicted_timeser <- ts(predict(lmfit), start=c(2011,1),end = c(2014,12), frequency=12)
predicted_timeser
plot_smoothed_series(2,total_time_series,"APAC Consumer Qty from Jan 2011 to Dec 2014","Quantity")
lines(predicted_timeser, col="red",lwd=2)

# check for noise
residue <- predicted_timeser - total_time_series


# Forecasting using arima
arima_forecast <- forecast.Arima(arima_model_APAC_Cons_Qty,h = 6)
arima_residue <- ts(arima_forecast$mean, start = c(2014,7), end=c(2014,12),frequency = 12) - ts_APAC_Cons_test_qty
#-----------------------------------------------------
# Mape value for regression model
mape_regression <-  mape(residue[43:48], ts_APAC_Cons_test_qty) # 10.60%
mape_regression
# Mape value for ARIMA
mape_arima <- mape(arima_residue, ts_APAC_Cons_test_qty) # 26.5%
mape_arima

#-----------------------------------------------------
# Choosing Regression as the final model
# Predicting from Jan 2015 and Jun 2015
APAC_Cons_Qty_pred <- forecast(lmfit,h=6)
APAC_Cons_Qty_pred

# ---------------------------------------------
#                     APAC Corporate
# --------------------------------------------
# Testing sales
total_time_series <- ts(c(ts_APAC_Corp,ts_APAC_Corp_test),start=c(2011,1),end = c(2014,12), frequency=12)
lmfit <-tslm(total_time_series ~ trend + season, data = total_time_series)
predicted_timeser <- ts(predict(lmfit), start=c(2011,1),end = c(2014,12), frequency=12)
predicted_timeser
plot_smoothed_series(2,total_time_series,"APAC Corporate Sales from Jan 2011 to Dec 2014","Sales")
lines(predicted_timeser, col="red")

# check for noise
residue <- predicted_timeser - total_time_series


# Forecasting using arima
arima_forecast <- forecast.Arima(arima_model_APAC_Corp_Sales,h = 6)
arima_residue <- ts(arima_forecast$mean, start = c(2014,7), end=c(2014,12),frequency = 12) - ts_APAC_Corp_test
#-----------------------------------------------------

# Mape value for regression model
mape_regression <-  mape(residue[43:48], ts_APAC_Corp_test) # 15.00%
mape_regression
#Mape value for ARIMA
mape_arima <- mape(arima_residue, ts_APAC_Corp_test) # 28.0%
mape_arima

#-----------------------------------------------------

# Choosing Regression as the final model
# Predicting from Jan 2015 and Jun 2015
APAC_Corp_Sales_pred <- forecast(lmfit,h=6)
APAC_Corp_Sales_pred

# Testing Quantity
total_time_series <- ts(c(ts_APAC_Corp_Qty,ts_APAC_Corp_test_qty),start=c(2011,1),end = c(2014,12), frequency=12)
lmfit <-tslm(total_time_series ~ trend + season, data = total_time_series)
predicted_timeser <- ts(predict(lmfit), start=c(2011,1),end = c(2014,12), frequency=12)
predicted_timeser
plot_smoothed_series(2,total_time_series,"APAC Corporate from Jan 2011 to Dec 2014","Quantity")
lines(predicted_timeser, col="red",lwd=2)

# check for noise
residue <- predicted_timeser - total_time_series


# Forecasting using arima
arima_forecast <- forecast.Arima(arima_model_APAC_Corp_Qty,h = 6)
arima_residue <- ts(arima_forecast$mean, start = c(2014,7), end=c(2014,12),frequency = 12) - ts_APAC_Corp_test_qty
#-----------------------------------------------------

# Mape value for regression model
mape_regression <-  mape(residue[43:48], ts_APAC_Corp_test_qty) # 8.13%
mape_regression
#Mape value for ARIMA
mape_arima <- mape(arima_residue, ts_APAC_Corp_test_qty) # 24.2%
mape_arima
#-----------------------------------------------------
# Choosing Regression as the final model
# Predicting from Jan 2015 and Jun 2015
APAC_Corp_Qty_pred <- forecast(lmfit,h=6)
APAC_Corp_Qty_pred

# ---------------------------------------------
#                     EU Corporate
# --------------------------------------------
# Testing sales
total_time_series <- ts(c(ts_EU_corp,ts_EU_corp_test),start=c(2011,1),end = c(2014,12), frequency=12)
lmfit <-tslm(total_time_series ~ trend + season, data = total_time_series)
predicted_timeser <- ts(predict(lmfit), start=c(2011,1),end = c(2014,12), frequency=12)
predicted_timeser
plot_smoothed_series(2,total_time_series,"EU Corporate Sales from Jan 2011 to Dec 2014","Sales")
lines(predicted_timeser, col="red")

# check for noise
residue <- predicted_timeser - total_time_series


# Forecasting using arima
arima_forecast <- forecast.Arima(arima_model_EU_Corp_Sales,h = 6)
arima_residue <- ts(arima_forecast$mean, start = c(2014,7), end=c(2014,12),frequency = 12) - ts_EU_corp_test
#-----------------------------------------------------

# Mape value for regression model
mape_regression <-  mape(residue[43:48], ts_EU_corp_test) # 22.07%
mape_regression
#Mape value for ARIMA
mape_arima <- mape(arima_residue, ts_EU_corp_test) # 37.0%
mape_arima
#-----------------------------------------------------
# Choosing Regression as the final model
# Predicting from Jan 2015 and Jun 2015
EU_corp_Sales_pred <- forecast(lmfit,h=6)
EU_corp_Sales_pred

#Testing Quantity
total_time_series <- ts(c(ts_EU_corp_Qty,ts_EU_corp_test_qty),start=c(2011,1),end = c(2014,12), frequency=12)
lmfit <-tslm(total_time_series ~ trend + season, data = total_time_series)
predicted_timeser <- ts(predict(lmfit), start=c(2011,1),end = c(2014,12), frequency=12)
predicted_timeser
plot_smoothed_series(2,total_time_series,"EU Corporate Qty from Jan 2011 to Dec 2014","Quantity")
lines(predicted_timeser, col="red",lwd=1.5)

# check for noise
residue <- predicted_timeser - total_time_series


# Forecasting using arima
arima_forecast <- forecast.Arima(arima_model_EU_Corp_Qty,h = 6)
arima_residue <- ts(arima_forecast$mean, start = c(2014,7), end=c(2014,12),frequency = 12) - ts_EU_corp_test_qty
#-----------------------------------------------------

# Mape value for regression model
mape_regression <-  mape(residue[43:48], ts_EU_corp_test_qty) # 28.69%
mape_regression
# Mape value for ARIMA
mape_arima <- mape(arima_residue, ts_EU_corp_test_qty) # 47.9%
mape_arima
#-----------------------------------------------------
# Choosing Regression as the final model
# Predicting from Jan 2015 and Jun 2015
EU_corp_Qty_pred <- forecast(lmfit,h=6)
EU_corp_Qty_pred

# ---------------------------------------------
#                     LATAM Consumer
# ---------------------------------------------

# Test Sales
total_time_series <- ts(c(ts_LATAM_cons,ts_LATAM_cons_test),start=c(2011,1),end = c(2014,12), frequency=12)
lmfit <-tslm(total_time_series ~ trend + season, data = total_time_series)
predicted_timeser <- ts(predict(lmfit), start=c(2011,1),end = c(2014,12), frequency=12)
predicted_timeser
plot_smoothed_series(2,total_time_series,"LATAM Consumer Sales from Jan 2011 to Dec 2014","Sales")
lines(predicted_timeser, col="red")

# check for noise
residue <- predicted_timeser - total_time_series


# Forecasting using arima
arima_forecast <- forecast.Arima(arima_model_LATAM_Cons_Sales,h = 6)
arima_residue <- ts(arima_forecast$mean, start = c(2014,7), end=c(2014,12),frequency = 12) - ts_LATAM_cons_test
#-----------------------------------------------------
# Mape value for regression model
mape_regression <-  mape(residue[43:48], ts_LATAM_cons_test) # 28.69%
mape_regression

# Mape value for ARIMA
mape_arima <- mape(arima_residue, ts_LATAM_cons_test) # 32.3%
mape_arima

#-----------------------------------------------------
# Choosing Regression as the final model
# Predicting from Jan 2015 and Jun 2015
LATAM_cons_Sales_pred <- forecast(lmfit,h=6)
LATAM_cons_Sales_pred

# Test Quantity
total_time_series <- ts(c(ts_LATAM_cons_Qty,ts_LATAM_cons_test_qty),start=c(2011,1),end = c(2014,12), frequency=12)
lmfit <-tslm(total_time_series ~ trend + season, data = total_time_series)
predicted_timeser <- ts(predict(lmfit), start=c(2011,1),end = c(2014,12), frequency=12)
predicted_timeser
plot_smoothed_series(2,total_time_series,"LATAM Consumer Qty from Jan 2011 to Dec 2014","Quantity")
lines(predicted_timeser, col="red",lwd=2)

# check for noise
residue <- predicted_timeser - total_time_series


# Forecasting using arima
arima_forecast <- forecast.Arima(arima_model_LATAM_Cons_Qty,h = 6)
arima_residue <- ts(arima_forecast$mean, start = c(2014,7), end=c(2014,12),frequency = 12) - ts_LATAM_cons_test_qty
#-----------------------------------------------------

# Mape value for regression model
mape_regression <-  mape(residue[43:48], ts_LATAM_cons_test_qty) # 14.23%
mape_regression
# Mape value for ARIMA
mape_arima <- mape(arima_residue, ts_LATAM_cons_test_qty) # 45%
mape_arima

#-----------------------------------------------------
# Choosing Regression as the final model
# Predicting from Jan 2015 and Jun 2015
LATAM_cons_Qty_pred <- forecast(lmfit,h=6)
LATAM_cons_Qty_pred
