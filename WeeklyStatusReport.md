# Weekly Status Reports
Weekly Status Reports for Casey Owen's Capstone Project for Tufts University's Master's in Data Science (MSDS)

## Contents
- [Week 1 05/29-06/05](#week-1)

## Status Reports

### Week 1
05/29 - 06/05

**Research Potential Project Ideas**
- I knew I wanted to do something related to energy/HVAC/renewable energy, so I searched the internet for the best publicly available datasets and compiled them in my notes
- I also put together a few initial ideas (in no particular order)
	- Idea 1 - **Optimize SEP grant funding distribution**
		- State funding from the DOE for energy programs via a specific program (SEP Formula Grants)  is given according to a formula proportional to energy use and total population.  However, there are more factors that go into what makes a good potential project than just these things. Get data on past projects that have been done with this grant funding and look at how their success varies by different features. Based on this, what is the most efficient way of distributing this funding?
		- Difficulty - this data will be hard to come by and may not be robust enough to make these predictions even if it exists. Does not seem publicly available and I have no relevant connections. Scrapping idea for now
	- Idea 2 - **Website dashboard of "what energy project is best for my home" using ResStock Dataset** 
		- There is a great data set called ResStock which looks at a large modelled sample of buildings that are intended to represent the whole of the US's residential building stock with tons of features per building. The data include the results of physics simulations that predict these building's energy based on their features, as well as the impact various measures (insulation, replace certain equipment, etc.) would have on these simulation.
			- Could use this data heavily to make a website tool that allows users to put in info about their building, and give a ranked list of what projects are most likely to save them the most money and emissions, based only on their building info
			- Allow people to see both energy savings and emissions savings on their projects
			- Ideally include confidence intervals on result, as well as project costs
			- Users might progressively enter more information about their home to get a more accurate estimate
			- Use information theory to suggest which features to prompt them to enter - which features that could be entered would move the needle the most on the energy estimates?
			- Possibly add solar installation as a measure myself - not in dataset, could probably infer info from somewhere else
		- Advantages
			- May be an actual useful tool that people need/use
			- I don't see something similar on the internet - somewhat novel
			- most/all data readily available
			- uses my existing work experience/expertise
		- Disadvantages
			- Dataset is very large so may be technically difficult to make site responsive
			- Current iteration does not really use machine learning or any advanced technical skills
			- Much of the work may just be creating the website - not really "Data science"?
			- Brainstorming a couple technical complexities to add to this may solve these problem
	- Idea 3 - **HVAC equipment anomaly detection**
		- Use equipment level data from the sensors of various equipment (AHUs for example) to do anomaly detection - detect when sensors have gone down, or when the equipment is malfunctioning
		- I found a dataset of synthetic building operations data I could use -  but hard to say how representative it truly is since it is synthetic
		- I also have a connection through my previous employer to an equipment monitoring company that would have a lot of this data - hard to say if they would be willing to let me use it. I am not aware of it, but its possible this was something they were already working on
		- Advantages
			- Anomaly detection is something I don't know about so it would be an ML learning opportunity
			- Uses my expertise in HVAC equipment data
			- Existing body of work on this subject that I can learn from
		- Disdvantages
			- Uncertain if I will get the data I want
			- Will want to brainstorm novel angles to take that haven't already been done
- Fails of the week
	- Did not settle on a specific idea yet
- Successes of the week
	- Came up with several ideas and compiled a lot of datasets
- Difficulties of the week
	- Coming up with a project idea that is the correct technical difficulty, related to my work experience, and answers an interesting question
- Goals for next week:
	- Meet with professor and settle on a specific idea, then complete project proposal and requirements specification
