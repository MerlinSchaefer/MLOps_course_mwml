

# Product

## Overview

### Background

**Set the scene for what we're trying to do through a customer-centric approach:**

- Customer: ML Devs and researchers
- Goal: stay up-to-date on ML content for work, knowledge etc.
- Pains: too much uncategorized content scattered around the internet
- Gains: a central location with categorized content from trusted 3rd party sources


|Assumption|Reality|     Reason    |
|----------|-------|----------------|
|  Assume we have customers on our platform already. |  Doesn't matter what you build if no one is there to use it.   |  This is a course on ML, not cold start growth. |
|Customers want to stay up-to-date with ML content. | Thorough customer studies required to confirm this.  |       We need a task that we can model in this course.      |


### Value Proposition

Propose the value we can create through a product-centric approach:

- Product: service that discovers and cate ML content from popular sources
- Alleviates: timely display of categorized content for customers to discover
- Advantages: customers only have to visit our product to stay up-to-date

(Papers, Videos, Articles, Repos, Courses ) -> (Classified and displayed for discovery)

### Objectives

Breakdown the product into key objectives to focus on.

- Allow customers to add and categorize their own projects.
- Discover ML content from trusted sources to bring into our platform.
- **Classify incoming content (with high precision) for our customers to easily discover. [OUR FOCUS]**
- Display categorized content on our platform (recent, popular, recommended, etc.)

|Assumption|Reality|    Reason         |
|----------|-------|--------|
|  Assume  we have a pipeline that delivers ML content from popular sources (Reddit, Twitter, etc.).  |  We would have to build this as a batch service and is not trivial.   | This is a course on ML, not batch web scraping. |

## Solution

**Describe the solution required to meet our objectives, including it's core features, integration, alternatives, constraints and what's out-of-scope.**

Here:

Develop a model that can classify the incoming content so that it can be organized by category on our platform.

**Core features:**

- ML service that will predict the correct categories for incoming content. [OUR FOCUS]
- user feedback process for incorrectly classified content.
- workflows to categorize content that the service was incorrect about or not as confident in.
- duplicate screening for content that already exists on the platform.
  
**Integrations:**

- categorized content will be sent to the UI service to be displayed.
- classification feedback from users will sent to labeling workflows.

**Alternatives:**

- allow users to add content manually (bottleneck).
  
**Constraints:**

- maintain low latency (>100ms) when classifying incoming content. [Latency]
- only recommend tags from our list of approved tags. [Security]
- avoid duplicate content from being added to the platform. [UI/UX]
  
**Out-of-scope:**

- identify relevant tags beyond our approved list of tags.
- using full-text HTML from content links to aid in classification.
- interpretability for why we recommend certain tags.
- identifying multiple categories (see dataset section for details).

### Feasibility:

How feasible is our solution and do we have the required resources to deliver it (data, $, team, etc.)?

**We have a dataset of ML content that our users have manually added to the platform. We'll need to assess if it has the necessary signals to meet our objectives.**

e.g.
```json 
{
    "id": 443,
    "created_on": "2020-04-10 17:51:39",
    "title": "AllenNLP Interpret",
    "description": "A Framework for Explaining Predictions of NLP Models",
    "tag": "natural-language-processing"
}
```

|Assumption|Reality|    Reason         |
|----------|-------|--------|
|This dataset is of high quality because they were added by actual users. | Need to assess the quality of the labels, especially since it was created by users! | The dataset is of good quality but we've left some errors in there so we can discover them during the evaluation process. |

# Engineering

**HOW can we engineer our approach for building the product**

## Data

**Describe the training and production (batches/streams) sources of data.**

training:

- access to input data and labels for training.
- information on feature origins and schemas.
- was there sampling of any kind applied to create this dataset?
- are we introducing any data leaks?

production:

- access to timely batches of ML content from scattered sources (Reddit, Twitter, etc.)
- how can we trust that this stream only has data that is consistent with what we have historically seen?

|Assumption|Reality|    Reason         |
|----------|-------|--------|
|ML stream only has ML relevant content.| Filter to remove spam content from these 3rd party streams| Would require us to source relevant data and build another model. |

### Labeling

Describe the labeling process and how we settled on the features and labels.

- Labeling: labeled using categories of machine learning (a subset of which our platform is interested in).

- Features: text features (title and description) to provide signal for the classification task.

- Labels: reflect the content categories we currently display on our platform:
```python
['natural-language-processing',
 'computer-vision',
 'mlops',
  ...
 'other']
 ```

 |Assumption|Reality|    Reason         |
|----------|-------|--------|
|Content can only belong to one category (multiclass).|Content can belong to more than one category (multilabel).|For simplicity and many libraries don't support or complicate multilabel scenarios.|

## Evaluation

**Before we can model our objective, we need to be able to evaluate how we’re performing.**

### Metrics

One of the hardest challenges with evaluation is tying our core objectives (may be qualitative) with quantitative metrics that our model can optimize on.

- We want to be able to classify incoming data with high precision so we can display them properly. For the projects that we categorize as other, we can recall any misclassified content using manual labeling workflows. We may also want to evaluate performance for specific classes or slices of data.
  
- Offline evaluation: We'll be using the historical dataset for offline evaluation. We'll also be creating slices of data that we want to evaluate in isolation.

- Online evaluation:
  - manually label a subset of incoming data to evaluate periodically.
  - asking the initial set of users viewing a newly categorized content if it's correctly classified.
  - allow users to report misclassified content by our model.

- It's important that we measure real-time performance before committing to replace our existing version of the system. 
  - Internal canary rollout, monitoring for proxy/actual performance, etc.
  - Rollout to the larger internal team for more feedback.
  - A/B rollout to a subset of the population to better understand UX, utility, etc.
  - **Not all releases have to be high stakes and external facing, you can always include internal releases to gather feedback and iterate**

## Modeling

While the specific methodology we employ can differ based on the problem, there are core principles we always want to follow:

- **End-to-end utility**: the end result from every iteration should deliver minimum end-to-end utility so that we can benchmark iterations against each other and plug-and-play with the system.
  
- **Manual before ML**: incorporate deterministic components where we define the rules before using probabilistic ones that infer rules from data → baselines.
  
- **Augment vs. automate**: allow the system to supplement the decision making process as opposed to making the final decision.

- **Internal vs. external**: not all early releases have to be end-user facing. We can use early versions for internal validation, feedback, data collection, etc.

- **Thorough**: every approach needs to be well tested (code, data + models) and evaluated, so we can objectively benchmark different approaches.

### Feedback

How do we receive feedback on our system and incorporate it into the next iteration? This can involve both human-in-the-loop feedback as well as automatic feedback via monitoring, etc.

- enforce human-in-loop checks when there is low confidence in classifications.
- allow users to report issues related to misclassification.

>**Attention**
> *Always return to the value proposition*
> 
> While it's important to iterate and optimize the internals of our workflows, it's even more important to ensure that our ML systems are actually making an impact. We need to constantly engage with stakeholders (management, users) to iterate on why our ML system exists.


# Project Management

**WHO & WHEN: organizing all the product requirements into manageable timelines so we can deliver on the vision.**

## Team

Which teams and specific members from those teams need to be involved in this project? It’s important to consider even the minor features so that everyone is aware of it and so we can properly scope and prioritize our timelines. Keep in mind that this isn’t the only project that people might be working on.

## Deliverables

We need to break down all the objectives for a particular release into clear deliverables that specify the deliverable, contributors, dependencies, acceptance criteria and status. This will become the granular checklist that our teams will use to decide what to prioritize.

## Timeline

This is where the project scoping begins to take place. Often, the stakeholders will have a desired time for release and the functionality to be delivered. There will be a lot of back and forth on this based on the results from the feasibility studies, so it's very important to be thorough and transparent to set expectations.