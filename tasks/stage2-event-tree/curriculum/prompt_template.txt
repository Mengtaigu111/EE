You are an expert in scientific information extraction.
Read the full abstract for context, but extract only the scientific events whose
TRIGGER occurs in the TARGET SENTENCE. Arguments may appear anywhere in the
full abstract.

Each event is a tree: the event type is the root, the trigger is the first child,
and each argument is a (Role "text") child. Order events by trigger position in
the target sentence. Inside an event: Trigger first, then roles in the schema
order below; several arguments of the same role go in text order.

If the target sentence has no event, output (ROOT).

Appendix F.2 exact event schema (event type: full name, template, roles):
PUR (Purpose)
Template: In order to deal with <Aim:arg1> under <Condition:arg2> circumstance on <Dataset:arg3> datasets.
Roles: Aim, Condition, Dataset
ITT (IntroduceTarget)
Template: <Target:arg1> is the abstract research target under <Condition:arg2> circumstance on <Dataset:arg3> datasets in this paper.
Roles: Target, Condition, Dataset
RWS (RelatedWorkStep)
Template: Previously <Subject:arg1> on <Target:arg2> are mostly based on <BaseComponent:arg3> with <TriedComponent:arg4> under <Condition:arg5> circumstance on <Dataset:arg6> datasets.
Roles: Subject, Target, BaseComponent, TriedComponent, Condition, Dataset
RWF (RelatedWorkFault)
Template: Aiming to <Target:arg1>, to <Extent:arg5> degree, <Concern:arg2> has some <Fault:arg6> faults under <Condition:arg3> circumstance on <Dataset:arg4> datasets.
Roles: Target, Extent, Concern, Fault, Condition, Dataset
PRP (Propose)
Template: In this paper, <Proposer:arg1> propose <Content:arg2> for <Target:arg3>.
Roles: Proposer, Content, Target
WKS (WorkStatement)
Template: <Researcher:arg1> report <Content:arg2> under <Condition:arg3> circumstance on <Dataset:arg4> datasets for <Target:arg5>.
Roles: Researcher, Content, Condition, Dataset, Target
MDS (MethodStep)
Template: Our approach adopt <BaseComponent:arg1> with <TriedComponent:arg2> under <Condition:arg3> circumstance on <Dataset:arg4> datasets for <Target:arg5>.
Roles: BaseComponent, TriedComponent, Condition, Dataset, Target
FIN (Finding)
Template: In experiments, <Finder:arg1> find or demostrate findings that <Content:arg2>.
Roles: Finder, Content
CMP (ExperimentCompare)
Template: Experimental results show that the <Metrics:arg6> of <Arg1:arg1> is <Extent:arg2> <Result:arg3> than <Arg2:arg4> under <Condition:arg5> circumstance on <Dataset:arg7> datasets.
Roles: Metrics, Arg1, Extent, Result, Arg2, Condition, Dataset
FAC (OutcomeFact)
Template: Experimental results show that <Subject:arg1> can <Extent:arg2> provide <Object:arg3> for <Target:arg4> under <Condition:arg5> circumstance on <Dataset:arg6> datasets because <Reason:arg7> reasons.
Roles: Subject, Extent, Object, Target, Condition, Dataset, Reason

Full Abstract:
[S1] <sentence 1>
[S2] <sentence 2>
...

TARGET SENTENCE [S{k}]:
<target sentence>

Event forest:
