export type Section = {
  id: string;
  title: string;
  paras?: string[];
  items?: { lead: string; text: string }[];
  after?: string[];
  figures?: { id: string; title: string; caption: string }[];
  table?: { head: string[]; rows: string[][]; note?: string; highlight?: number };
};
export type Lens = 'ai' | 'ml' | 'data';
export const lenses: { id: Lens; label: string }[] = [
  { id: 'ai', label: 'AI & Evaluation' },
  { id: 'ml', label: 'Machine Learning' },
  { id: 'data', label: 'Data & Analytics' },
];
export type Link = { label: string; href: string };
export type Project = {
  slug: string;
  title: string;
  kind: string;
  category: 'Research' | 'Product' | 'AI and machine learning' | 'Data and analytics';
  year: string;
  seoTitle?: string;
  summary: string;
  lead: string;
  stack: string[];
  metrics?: { value: string; label: string }[];
  kpis: { value: string; label: string }[];
  links: Link[];
  sections: Section[];
  credits?: { text: string; href: string }[];
  limits: string[];
  lenses: Lens[];
  featured?: boolean;
};

const mail = 'mailto:contact@karimabbas.com?subject=';

export const projects: Project[] = [
  {
    slug: 'armadillo',
    title: 'ARMADILLO',
    seoTitle: 'ARMADILLO: midline loss for 3D landmark segmentation',
    kind: 'Masters research',
    category: 'Research',
    lenses: ['ml'],
    year: '2025',
    summary:
      'A novel loss function for segmenting thin anatomical landmarks on 3D liver models, published at MIUA 2025.',
    lead:
      'Part of a group research project on lining up a 3D model of the liver with live surgical video. I wrote the paper on the first step, finding landmarks on the 3D model. Our loss function cuts the average landmark error by 37% compared with leading state-of-the-art research, from about 26 mm to about 17 mm.',
    stack: ['Python', 'Deep learning', 'PointNet++', 'Point clouds', 'Computer vision'],
    metrics: [{ value: '37%', label: 'lower landmark error than leading state-of-the-art research (26.2 mm down to 16.6 mm)' }],
    kpis: [
      { value: '37%', label: 'lower error than leading state-of-the-art research' },
      { value: '48%', label: 'lower than the next-best state-of-the-art method' },
      { value: '16.6 mm', label: 'mean error on unseen livers' },
    ],
    links: [
      { label: 'Read the paper', href: 'https://doi.org/10.1007/978-3-031-98691-8_16' },
      { label: 'Code', href: 'https://github.com/ARMADILLO-VISION/midline-loss' },
      { label: 'Companion paper', href: 'https://doi.org/10.1007/978-3-031-98694-9_15' },
    ],
    featured: true,
    sections: [
      {
        id: 'background',
        title: 'Where it started',
        paras: [
          'In keyhole liver surgery the surgeon works from a camera feed. A 3D model built from an earlier scan can be laid over that feed, but only after the two have been lined up, and lining them up needs landmarks marked on both. Marking them by hand is slow and varies from person to person.',
          'ARMADILLO set out to take that step away. The group built the whole chain: landmarks on the 3D model, landmarks in the video, the registration between them, and a viewer to inspect the results.',
        ],
      },
      {
        id: 'idea',
        title: 'The idea behind the midline loss',
        paras: [
          'The two landmarks we looked at, the falciform ligament and the liver ridge, are thin lines on the surface. Earlier approaches used MeshCNN, graph networks and PointNet-style models, and tended to predict regions that were too wide or too sparse. That is partly down to the loss. A standard loss only asks whether each point is labelled correctly, so a fat blob around the right spot costs very little.',
          'The midline loss adds a geometric check. It fits a central axis through the predicted points, snaps that axis to the points in a way that stays differentiable, then penalises two things: points that sit far from the axis, and an axis that drifts from the true centreline. Predictions come out thin and in the right place. We built it on PointNet++ and trained on a combined set of about 300 liver models.',
        ],
      },
      {
        id: 'results',
        title: 'How it compared',
        paras: ['Error is the mean Chamfer distance in millimetres, so lower is better. All rows are on the unseen P2ILF test set.'],
        table: {
          head: ['Method', 'Ridge', 'Ligament', 'Mean'],
          rows: [
            ['PointNet++ with midline loss', '19.70', '13.46', '16.58'],
            ['UCL, leading state of the art', '27.97', '24.47', '26.22'],
            ['NCT, next-best state of the art', '27.19', '36.38', '31.79'],
            ['PointNet++ with standard weighted cross-entropy, same training', '23.72', '60.26', '41.99'],
          ],
          highlight: 0,
        },
        after: [
          'The result I found most interesting is about generalisation. On the held-out slice of the larger dataset, plain cross-entropy scored better than the midline loss (12.7 mm against 16.6 mm). Moved to livers from a different source, it fell to 42 mm while the midline model held at 16.6 mm. My reading is that the constraint stops the network from learning quirks of one dataset.',
          'Across the whole pipeline the group report records a 30% relative gain in 2D segmentation precision, 36% better 3D segmentation distance, and 31% lower reprojection error than leading earlier work.',
        ],
      },
      {
        id: 'team',
        title: 'Working in a group',
        paras: [
          'ARMADILLO was a seven-person project and every member is on the paper, and it won the School of Computer Science’s best MEng project prize. I am first author on the midline loss work. The 2D segmentation and registration were done by the rest of the group, and the second MIUA paper covers the 2D segmentation side.',
        ],
      },
    ],
    limits: [
      'The P2ILF test set is two patients. I would want more before leaning on the generalisation result.',
      'This is research code. It has not been used in a clinical setting.',
    ],
  },
  {
    slug: 'lapbylap',
    title: 'LapByLap',
    seoTitle: 'LapByLap: F1 Fantasy race simulation and team optimiser',
    kind: 'Live product',
    category: 'Product',
    lenses: ['ml', 'data'],
    year: '2025',
    summary:
      'An F1 Fantasy app that simulates each race thousands of times and picks the best team within the budget.',
    lead:
      'F1 Fantasy gives you a budget and asks for five drivers and two constructors. LapByLap runs each race weekend 10,000 times from real timing data, then finds the team with the best expected points for the amount of risk you want to take. Over the 2026 season its predictions have ranked drivers correctly 82% of the time. It is live and I maintain it myself.',
    stack: ['Python', 'FastAPI', 'React', 'PostgreSQL', 'Three.js', 'Docker', 'NumPy', 'Pandas', 'PuLP'],
    metrics: [
      { value: '82%', label: 'directional accuracy, 2026 season' },
      { value: '90%', label: 'top-5 hit rate, 2026 season' },
    ],
    kpis: [
      { value: '82%', label: 'directional accuracy' },
      { value: '90%', label: 'top-5 hit rate' },
      { value: '0.78', label: 'Spearman rank correlation' },
    ],
    links: [
      { label: 'Open the app', href: 'https://lapbylap.karimabbas.com' },
      { label: 'Methodology', href: 'https://lapbylap.karimabbas.com?page=methodology' },
      { label: 'See London cycle hire forecasting', href: '/projects/cycle-hire-forecasting' },
    ],
    featured: true,
    sections: [
      {
        id: 'why',
        title: 'Why simulate',
        paras: [
          'Most Fantasy advice is built on last race’s results. That hides how much a race can swing. A cheap driver with a good chance of finishing in the points can be worth more than an expensive one who retires every few weekends.',
          'So LapByLap treats a race as a spread of outcomes and picks a team against the whole spread.',
        ],
      },
      {
        id: 'built',
        title: 'How it works',
        items: [
          { lead: 'Driver models.', text: 'Timing data from FastF1 gives each driver a qualifying and race pace distribution, a tyre wear rate and a chance of not finishing.' },
          { lead: 'Race simulation.', text: 'Each run plays the race lap by lap, with tyre wear, fuel burn, traffic, safety cars and retirements adding noise to the order.' },
          { lead: 'Team selection.', text: 'A linear program (PuLP) picks the lineup within the budget and team limits. It scores teams on expected points minus a penalty for variance, so risk is a setting you can turn.' },
          { lead: 'The app.', text: 'A FastAPI backend with PostgreSQL, a React front end with a Three.js view, packaged with Docker and running on its own domain.' },
        ],
      },
      {
        id: 'features',
        title: 'What the app shows',
        items: [
          { lead: 'Race view.', text: 'Simulated finishing positions for each driver, including how the position changes lap by lap and how wide the range of outcomes is.' },
          { lead: 'Value charts.', text: 'Price against expected points for drivers and constructors, so you can see who is cheap for what they score.' },
          { lead: 'Risk against reward.', text: 'A quadrant of expected points against variance, which is where the risk setting comes from.' },
          { lead: 'Methodology page.', text: 'The modelling is written up in the app itself, so the numbers can be checked.' },
        ],
      },
      {
        id: 'picking',
        title: 'Choosing a team',
        paras: [
          'You pick the budget cap and how much risk you are willing to carry. The optimiser then returns the lineup with the best score under the game’s real rules: five drivers, two constructors, a limit on drivers from one team, and the cost ceiling. Move the risk setting and the lineup changes, which makes the trade-off easy to see.',
        ],
      },
      {
        id: 'results',
        title: 'How it has done',
        paras: ['Each race I compare the model’s predictions with what actually happened. These are the numbers for the 2026 season, and what each one measures.'],
        table: {
          head: ['Metric', 'In plain terms', '2026 season'],
          rows: [
            ['Directional accuracy', 'Pick any two drivers: how often the model correctly says who scores more. 50% is a coin flip.', '82%'],
            ['Top-5 hit rate', 'Of the five drivers the model rates highest, the share that finish in the real top 10.', '90%'],
            ['Spearman correlation', 'How well the predicted ranking of drivers matches the real one. 1 is identical, 0 is no relationship.', '0.78'],
            ['MAE', 'Typical miss on a single driver’s fantasy points, in points. Lower is better.', '10.4 average (6.8 to 18.5 by race)'],
            ['Bias', 'Whether the model leans high or low overall. Close to zero is what you want.', '-0.4 points'],
            ['80% coverage', 'How often the real result falls inside the model’s 80% range. Ideal is about 80%.', '79%'],
          ],
        },
      },
      {
        id: 'model',
        title: 'The modelling in a bit more detail',
        paras: [
          'Qualifying and race lap times are modelled as normal distributions fitted to each driver’s history. Simulated qualifying gaps set the grid. Race lap time then adds tyre degradation, fuel effect, a traffic penalty and safety-car effects, with each driver’s volatility taken from how consistent their pace has been.',
          'The optimiser maximises total expected points, adds credit for the best scorer on the team, and takes off a penalty for portfolio variance, normalised by cost.',
        ],
      },
    ],
    limits: [
      'Tyre strategy is simplified and weather is modelled only roughly.',
      'It needs a decent amount of history per driver, so rookies and mid-season swaps are weak spots.',
      'The figures cover a single season, so treat them as a snapshot rather than a guarantee.',
      'A wet or safety-car-heavy race can still land well outside the predicted range.',
    ],
  },
  {
    slug: 'individual-project',
    title: 'University course assistant',
    kind: 'Final-year project',
    category: 'AI and machine learning',
    lenses: ['ai', 'ml'],
    year: '2024',
    summary:
      'A study assistant for one university module, built with retrieval and fine-tuning on GPT-3.5. First Class (73%).',
    lead:
      'Retrieval-augmented generation is now a core part of my day job, and this is where I first built it. A general chatbot can explain a topic, but it does not know what your lecturer covered. This project gave GPT-3.5 access to a module’s lecture material and fine-tuned it on course-style questions, then checked whether students got better answers.',
    stack: ['Python', 'GPT-3.5', 'Retrieval-augmented generation', 'Fine-tuning', 'NLP'],
    kpis: [],
    links: [
      { label: 'Code', href: 'https://github.com/icekreem27/final-year-project-icekreem27' },
      { label: 'Ask for the report', href: mail + 'Course%20assistant%20report' },
    ],
    sections: [
      {
        id: 'problem',
        title: 'The question',
        paras: [
          'Students have module-specific questions and lecturers cannot answer them one at a time. Off-the-shelf models answer fluently but without the lecture context, so the answers can be generic or subtly wrong for the course.',
        ],
      },
      {
        id: 'built',
        title: 'What I built',
        items: [
          { lead: 'Retrieval.', text: 'Lecture material is indexed so the most relevant passages are pulled into each answer.' },
          { lead: 'Fine-tuning.', text: 'GPT-3.5 was fine-tuned on academic questions in the style of the course.' },
          { lead: 'Evaluation.', text: 'I compared it with the standard model using automatic scores such as ROUGE, and collected feedback from university students.' },
        ],
      },
      {
        id: 'results',
        title: 'What I found',
        paras: [
          'The enhanced assistant gave more specific answers than the standard model and stayed closer to the source lecture. Students who tried it said it would help their learning. I presented it at the University of Leeds Digital AI Summit, and the project was awarded a First (73%).',
        ],
      },
    ],
    limits: [
      'It was tested on one module with a small group of students, so the results may not carry over to other courses.',
      'The full metric tables are in the report, which I am happy to share.',
    ],
  },
  {
    slug: 'spotify',
    title: 'Spotify playlist recommendations',
    kind: 'Personal project',
    category: 'AI and machine learning',
    lenses: ['ml'],
    year: '2023',
    summary: 'Suggests new tracks for a playlist by comparing audio features with cosine similarity.',
    lead:
      'Give it a playlist and it suggests tracks that sound similar. I wanted hands-on practice with large datasets, and vector similarity was fresh from my final-year project.',
    stack: ['Python', 'Pandas', 'NumPy', 'Scikit-learn', 'Matplotlib', 'Seaborn'],
    kpis: [],
    links: [
      { label: 'Notebook and code', href: 'https://github.com/icekreem27/spotify-playlist-recommendation' },
      { label: 'Dataset', href: 'https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-19212020-600k-tracks/data' },
    ],
    sections: [
      {
        id: 'built',
        title: 'What I built',
        items: [
          { lead: 'Data preparation.', text: 'Cleaned a Kaggle dataset of Spotify tracks, filled or dropped missing values and put the audio features (tempo, energy, danceability and so on) on a common scale.' },
          { lead: 'Feature work.', text: 'Looked at how the features relate and used PCA to shrink them before comparing tracks.' },
          { lead: 'Recommender.', text: 'A function, generate_playlist_recos, scores every track outside the playlist by cosine similarity to the playlist’s tracks and returns the closest ones with a similarity score.' },
        ],
      },
    ],
    limits: [
      'I judged the output by listening, not with an offline metric.',
      'It only uses audio features. Connecting it to the Spotify API and trying learned models are the obvious next steps.',
    ],
  },
  {
    slug: 'cycle-hire-forecasting',
    title: 'London cycle hire forecasting',
    seoTitle: 'London cycle hire forecasting: probabilistic demand forecasts and a rebalancing test',
    kind: 'Personal project',
    category: 'Data and analytics',
    lenses: ['ml', 'data'],
    year: '2026',
    summary:
      'Hourly forecasts, with a range, for how many bikes leave 80 busy London docks, tested against data I locked away until the end.',
    lead:
      'Santander Cycles docks run empty or full, and a van can only visit so many. This project forecasts how many bikes will leave each of 80 busy docks in every hour of the next day, with a range around the number, then asks what that range is worth to someone sending vans. On spring 2026 data I kept untouched until every choice was fixed, the forecasts have 18% lower error than the fairest simple baseline.',
    stack: ['Python', 'LightGBM', 'Pandas', 'SciPy', 'Statsmodels', 'Conformal prediction', 'Backtesting', 'Simulation'],
    metrics: [{ value: '18%', label: 'lower forecast error than the fair baseline, on spring 2026 data locked away until the end' }],
    kpis: [
      { value: '18%', label: 'lower error than the fair baseline on the locked-away test' },
      { value: '1.17', label: 'trips per dock-hour typical miss over two years, against 1.33 for the baseline' },
      { value: '18.1M', label: 'trips in the TfL data behind it' },
    ],
    links: [
      // Placeholder until the repo is public.
      { label: 'Code (coming soon)', href: '#' },
      { label: 'See LapByLap', href: '/projects/lapbylap' },
    ],
    sections: [
      {
        id: 'why',
        title: 'The question',
        paras: [
          'An operator of a bike scheme sends vans to move bikes. A dock that runs empty loses a trip. A dock that runs full blocks a return. Those two costs are not the same, so the right number of bikes to leave at a dock is not simply the number you expect to be taken.',
          'So the forecast has to say how sure it is, not only what it expects. This project forecasts departures and returns for each dock for the next 24 hours, issued every morning at 06:00 UTC, and then tests whether knowing the spread changes the decision.',
          'It sits next to LapByLap. Both predict first and optimise second, but this one has real operating data and no Formula 1.',
        ],
      },
      {
        id: 'data',
        figures: [
          { id: 'demand', title: 'What demand looks like', caption: 'Each line is the average number of bikes leaving a dock in that hour of the day. I split the 80 docks into two groups of 40: morning docks, the busiest at the weekday commute, and weekend docks, the busiest on weekends. Each group is shown on weekdays and on weekends, so the dark solid line is a commuter dock on a weekday, with a sharp peak at 8am. The bars underneath show that 44% of dock-hours have no departures at all, which is why the model treats trips as counts.' },
        ],
        title: 'The data, and what is wrong with it',
        paras: [
          'The base is TfL’s public journey data: 18.1 million trips from June 2024 to May 2026, turned into hourly counts per dock. I picked 80 docks, 40 that are busy at the weekday morning peak and 40 that are busy at weekends, each open under one name for the whole period. Forecast weather comes from Open-Meteo, bank holidays from GOV.UK, school terms from Housemait and Tube strike dates from a list I compiled and checked against a TfL freedom of information response.',
          'Most of the work was finding where the data lies.',
        ],
        items: [
          { lead: 'Trips taken are not trips wanted.', text: 'An empty dock records no demand. The archive has no dock availability, so I score only docks that are rarely empty and say so beside every headline number. I also flag hours where a normally busy dock logged nothing, which is a floor on how much demand is hidden.' },
          { lead: 'A dead day.', text: 'On 5 August 2025 the whole network logged 188 trips against about 30,700 on a normal Tuesday. That is the data failing, not demand. It had been sitting in a scored window as zeros, so it is now missing, and so are the days that read it.' },
          { lead: 'Closed docks.', text: 'A dock silent for a day or more, when it normally logs plenty, is closed, not unpopular. 2,016 dock-hours match events I can name: the Hyde Park docks around Winter Wonderland, the Notting Hill Carnival, New Year at Waterloo. They are treated as missing.' },
          { lead: 'A feature that did nothing.', text: 'School-term dates for 2024/25 were missing, so I compiled them by hand from five borough calendars. The flag moved the score by 0.0003, with an interval spanning zero. It stays in because it costs nothing, but I do not credit it.' },
        ],
      },
      {
        id: 'model',
        title: 'How the model works',
        paras: [
          'The count of bikes leaving a dock in an hour is a small whole number: the average is about two, and 44% of hours are zero. That shape matters, because a model built to hit the middle of a smooth curve behaves badly on it.',
        ],
        items: [
          { lead: 'Step one, the average.', text: 'A LightGBM model, trained on all 80 docks together, predicts the expected count for each dock-hour. It reads the same hour in each of the last four weeks, the last day and the last six hours, the hour and weekday, bank holidays, school terms, Tube strikes and the weather forecast for that morning. It uses Poisson loss, the standard choice for counting events.' },
          { lead: 'Step two, the spread.', text: 'A negative binomial turns that average into a range. It is a distribution for counts that are usually near the average and sometimes burst. How bursty depends on how busy the dock is, so each volume level has its own setting, fitted by maximum likelihood on spare mornings.' },
          { lead: 'Step three, the safety margin.', text: 'A conformal step checks the range against recent mornings the model did not train on and nudges it wider or narrower, separately for each hour of the day and for busy and quiet docks. It targets 80%, and on whole numbers it lands higher, because a range has to round outward when 44% of outcomes are zero.' },
        ],
      },
      {
        id: 'testing',
        figures: [
          { id: 'robustness', title: 'Picking the docks without hindsight', caption: 'The same five models on the original 80 docks and on 80 re-picked from the first six months of data only, scored on the six later windows. The spring 2026 test is not used.' },
        ],
        title: 'How I tested it',
        paras: [
          'The test is a rolling backtest. Each forecast uses only data from before 06:00 UTC on its day, and the weather it sees is the forecast that was stored then, not what actually happened. There are nine windows of 28 mornings across two years, and a test fails if any feature can see the future.',
          'The fair baseline is the quantiles of the same hour over the last eight weeks, with no fitting. I added it after noticing that “same hour last week” was an easy bar: it scored half the error of my first model, and the real gap was a seventh of that. ARIMA, which had looked like progress against last week, turned out to be worse than the no-fit baseline.',
          'Spring 2026, which includes the April Tube strike, was kept locked. I opened it once, on 29 October 2026, after every model, setting and cost was fixed. The code refuses to run it a second time.',
          'I also checked my own dock selection. The 80 docks were first picked with the whole two years in view, which can flatter results. Re-picking them from the first six months only, and scoring on the six later windows, moved the count model’s mean pinball from 0.349 to 0.348 and left the ranking untouched. 65 of the 80 docks were the same.',
        ],
      },
      {
        id: 'results',
        figures: [
          { id: 'leaderboard', title: 'Five models, same dock-hours', caption: 'Lower is better. The dashed line is the fair baseline, the same hour over the last eight weeks with no fitting. The dots below show the gap to it with 95% intervals, resampling whole mornings. Negative is better.' },
          { id: 'forecast', title: 'One dock, one week', caption: 'The forecast against what happened. The shaded band is the model’s 10 to 90% range, issued each morning at 06:00 UTC for the next 24 hours. The model had not seen any of these days. It opens on Argyle Street, the busy dock with the lowest error relative to its volume that week, so it shows the model at its best. Every other dock is one click away.' },
          { id: 'coverage', title: 'How often the real count landed inside the range', caption: 'An 80% range on whole numbers has to round outward, so it holds more than 80%. The shape matters more than the level.' },
          { id: 'slices', title: 'Where the errors are, spring 2026', caption: 'Strike hours are 21 to 24 April 2026. The count model’s gain over the baseline is larger on strike and bank-holiday hours than on ordinary ones.' },
        ],
        title: 'How it did',
        paras: [
          'Lower is better in the first two rows. Mean pinball loss scores the whole range, not only the middle guess. All five models are scored on the same dock-hours.',
        ],
        table: {
          head: ['', 'Last week', 'Last 8 weeks', 'ARIMA', 'LightGBM', 'Count model'],
          rows: [
            ['Typical miss, trips per dock-hour (spring 2026)', '1.69', '1.45', '1.69', '1.32', '1.25'],
            ['Mean pinball, spring 2026', '0.846', '0.486', '0.580', '0.418', '0.399'],
            ['Mean pinball, nine earlier windows', '0.798', '0.448', '0.547', '0.389', '0.375'],
            ['Gap to the fair baseline, spring 2026', '+0.359', 'baseline', '+0.094', '−0.068', '−0.087'],
          ],
          note: 'Gaps are mean pinball minus the last-8-weeks baseline, so negative is better. The count model’s gap is −0.087 (−0.104 to −0.073) and 95% intervals come from resampling whole mornings.',
        },
        after: [
          'The ranking was the same in the nine earlier windows and on the locked-away spring: the count model is 16% better than the fair baseline in the first and 18% in the second. Strike and bank-holiday hours are harder for everyone, but its gain over the baseline is larger there, about 27% and 32% against 16% on ordinary hours.',
          'The range is not perfectly calibrated and I would rather say where. Across the nine windows the 8am peak was the weak spot, with the range holding 85% against 92% overall. In spring 2026 it did not show up. One window of 56 mornings is a result, not a fix.',
        ],
      },
      {
        id: 'gain',
        figures: [
          { id: 'ablation', title: 'One change at a time', caption: 'Change in mean pinball loss, negative is better. A whisker that crosses zero means no detectable effect.' },
        ],
        title: 'Where the gain came from',
        paras: [
          'My first count model beat plain LightGBM by 4%, but it differed in four ways at once. Before crediting the clever part, I changed one thing at a time.',
        ],
        table: {
          head: ['Change', 'Effect on mean pinball', 'Verdict'],
          rows: [
            ['Four weeks of history instead of one, quantile model', '−0.007 (−0.009 to −0.004)', 'Real'],
            ['Count distribution instead of quantile loss, same inputs', '−0.007 (−0.009 to −0.005)', 'Real'],
            ['Dropping forecast weather', '+0.004 (+0.001 to +0.008)', 'Helps a little'],
            ['Dropping the school-term flag', '+0.0003 (−0.0004 to +0.001)', 'No effect'],
          ],
          note: 'Negative is better. An interval that spans zero means no detectable effect.',
        },
        after: [
          'So about half the lead was just more history, and the other half was the count distribution. The model I thought was clever was less than half of the story, and I would rather know that.',
        ],
      },
      {
        id: 'decision',
        figures: [
          { id: 'explorer', title: 'Try the costs', caption: 'Each window is one dock and four hours. Scenarios change one input at a time from the base case, so not every pair of settings exists.' },
          { id: 'assumptions', title: 'What the test assumes', caption: 'Which inputs are sourced and which are my assumptions. The assumed ones are why the sweep is the result and the base case is not.' },
        ],
        title: 'What it is worth to someone sending vans',
        paras: [
          'The test: for each dock and a four-hour window, the operator picks how many bikes to have there at the start. Changing from half full costs a van stop. Policies are scored on what actually happened, with costs for a lost trip, a blocked return and a stop.',
          'Docks per station come from TfL’s live listing (13 to 55, average 29). A lost trip costs 1.65, the pay-as-you-go fare. Assume a van holds 18 bikes. The cost of a van stop, 10, is my arithmetic, since no UK operator publishes one, so I swept it from 3 to 50.',
        ],
        table: {
          head: ['Policy', 'Cost per window', 'Avoidable cost removed'],
          rows: [
            ['No rebalancing', '5.80', '0%'],
            ['Plan from last week’s counts as if exact', '5.78', '4%'],
            ['Plan from the model’s average as if exact', '5.55', '51%'],
            ['Plan against the whole forecast', '5.53', '54%'],
            ['Oracle that sees the future', '5.31', '100%'],
          ],
          note: 'Nine earlier windows, base costs. On the locked-away spring the same shares are 10%, 61% and 64%.',
          highlight: 3,
        },
        after: [
          'The clearest result is that the model beats last week’s counts by a lot. A noisy single number sends vans that were not needed, and was worse than doing nothing at the quiet docks. Having the forecast at all is where most of the gain is.',
          'The spread adds little in this setup. Planning against the whole forecast beats planning from the average by about 0.02 per window, 0.3% of total cost, and on spring 2026’s 56 mornings that gap is within noise. Even an oracle saves only about 8% of total cost, because most of the cost is the busiest third of docks, where four hours of demand exceeds the docks whatever anyone does.',
          'So I tried a harder version. A real operator has a van that can make only so many stops a morning, so the forecast has to choose which docks to visit.',
        ],
      },
      {
        id: 'budget',
        figures: [
          { id: 'budget', title: 'A limited van', caption: 'Pounds saved per morning across the 80 docks, against doing nothing, which costs about 460 a morning. Each policy ranks the docks by what it expects a visit to save, visits the top ones and is scored on what happened. Nine earlier windows. The spring 2026 test was not used for this.' },
        ],
        title: 'A limited van',
        paras: [
          'Each morning one van can make at most so many stops across the 80 docks. “Van already paid for” means a stop costs only the handling, 0.4 a bike. Otherwise a stop costs 10.',
        ],
        table: {
          head: ['Setup', 'Last week', 'Model average', 'Whole forecast', 'Oracle'],
          rows: [
            ['Stop costs 10, 3 stops a morning', '10.6', '15.9', '17.2', '25.1'],
            ['Van paid for, 4 stops a morning', '41.7', '49.1', '50.4', '63.4'],
            ['Van paid for, 16 stops a morning', '62.5', '71.7', '80.5', '133.6'],
            ['Van paid for, 32 stops a morning', '71.4', '78.1', '93.7', '161.4'],
          ],
          note: 'Pounds saved per morning. Whole forecast minus model average at 16 stops: −8.8 a morning (−10.9 to −6.6), resampling mornings.',
          highlight: 2,
        },
        after: [
          'Here the spread earns its place. Ranking docks by the whole forecast saves about 7 to 16% more than ranking by the average when a stop costs 10, and up to 24% more when a stop costs only the handling. The reason is the textbook one. A lost trip costs about twice a blocked return, so the right stocking level is well above the average, and a plan built on the average under-stocks.',
          'The honest summary is that the spread matters when moving a bike is cheap compared with a lost trip, and adds little when every stop is expensive. I would rather show where it helps and where it does not than pick the setup that flatters it. I also modelled departures and returns moving together, with a Gaussian copula. It matched the spread of real outcomes better and changed no decision, so the simple version stays.',
        ],
      },
    ],
    credits: [
      { text: 'Powered by TfL Open Data', href: 'https://tfl.gov.uk/corporate/terms-and-conditions/transport-data-service' },
      { text: 'Contains OS data © Crown copyright and database rights 2016', href: 'https://tfl.gov.uk/corporate/terms-and-conditions/transport-data-service' },
      { text: 'Geomni UK Map data © and database rights [2019]', href: 'https://tfl.gov.uk/corporate/terms-and-conditions/transport-data-service' },
      { text: 'Weather data by Open-Meteo.com', href: 'https://open-meteo.com/' },
      { text: 'School term dates from Housemait, CC BY 4.0', href: 'https://housemait.com/school-term-dates/data' },
    ],
    limits: [
      'Trips taken stand in for trips wanted. Empty docks hide demand, and the archive cannot show how much. I flag a floor, about 0.8% of departures, and score docks that are rarely empty.',
      'The 80 docks were first chosen with the whole two years in view. A re-pick using only the first six months gave the same scores, but it is still one network over two years, not a different city or period.',
      'The van-stop cost, the cost of a blocked return, the cost of moving a bike and the default start level are assumptions, not operator data, which is why the result is the sweep and not the base case.',
      'The decision test is one stop per dock per window, with no routing and a van limited to the bikes it holds.',
      'The spring 2026 test is one season of 56 mornings with one strike. Dock counts are today’s layout, not the layout on every day of the two years.',
    ],
  },
  {
    slug: 'airbnb-analysis',
    title: 'Airbnb market analysis',
    kind: 'Personal project',
    category: 'Data and analytics',
    lenses: ['data'],
    year: '2023',
    summary: 'Reshaping a large Airbnb dataset into a SQL database, then asking it ten questions.',
    lead:
      'The source data was one huge CSV covering over 250,000 listings in ten cities. I redesigned it as a set of related tables, loaded it into SQL, and wrote queries to answer ten questions about hosts, prices and reviews.',
    stack: ['SQL', 'MySQL', 'Python', 'Pandas', 'Tableau'],
    kpis: [
      { value: '250k+', label: 'listings across 10 cities' },
      { value: '5M+', label: 'reviews in the source data' },
      { value: '6', label: 'key tables after normalising' },
    ],
    links: [{ label: 'Code', href: 'https://github.com/icekreem27/Airbnb-SQL-Analysis' }],
    sections: [
      {
        id: 'built',
        title: 'What I did',
        items: [
          { lead: 'Schema.', text: 'Split the flat file into Host, Location, Property, ReviewScores, Listing and Review tables to remove repetition and make queries faster.' },
          { lead: 'Loading.', text: 'My first plan was to load everything and split it with SQL, which was slow and fiddly. It worked far better to build CSVs in pandas that already matched the final schema and import those.' },
          { lead: 'Reviews.', text: 'A short Python script matched listing IDs against the reviews file so only relevant reviews were kept.' },
          { lead: 'The questions.', text: 'Who the most active hosts are, how many are superhosts, which neighbourhoods have the most listings and what they cost, how room types and property types split, how review scores look, and how prices and review activity change over time.' },
        ],
      },
    ],
    limits: [
      'I built and cleaned everything on a 500-row subset before scaling, so the full dataset has not been through the whole pipeline.',
      'The Tableau dashboard was an early version when I wrote this up.',
    ],
  },
  {
    slug: 'f1-predictions',
    title: 'F1 Fantasy predictor',
    kind: 'Personal project',
    category: 'Data and analytics',
    lenses: ['ml', 'data'],
    year: '2024',
    summary: 'The first version of the Fantasy idea, a notebook that picks the best-value lineup. It became LapByLap.',
    lead:
      'Where LapByLap started: use the current season’s data to estimate the best-value Fantasy lineup for each race.',
    stack: ['Python', 'Pandas', 'Machine learning'],
    kpis: [],
    links: [{ label: 'See LapByLap', href: '/projects/lapbylap' }],
    sections: [
      {
        id: 'built',
        title: 'What it did',
        paras: [
          'It took race and pricing data for the season, estimated expected points for each driver and constructor, and returned a lineup per race weekend.',
          'It gave a single answer with no sense of how uncertain it was, and you could not change how much risk to take. Those two gaps are what LapByLap was built to fix.',
        ],
      },
    ],
    limits: ['Replaced by LapByLap and no longer maintained.'],
  },
];
