'''
All code must be in the my-submission folder (i.e. copy -r deep_nmmo/deep_nmmo/
into this folder).

Instructions on how to submit: https://gitlab.aicrowd.com/neural-mmo/neurips2022-nmmo-starter-kit

python tool.py test --startby=docker
python tool.py submit <unique-submission-name> --startby=docker
'''

from deep_nmmo.envs.team_based_env.teams.custom_team import CustomTeam

class Submission:
    team_klass = CustomTeam
    init_params = {'paths_to_agents_cls': 
                    {
                        '0': 'neurips2022nmmo.scripted.baselines.Herbalist',
                        '1': 'neurips2022nmmo.scripted.baselines.Melee',
                        '2': 'neurips2022nmmo.scripted.baselines.Range',
                        '3': 'neurips2022nmmo.scripted.baselines.Fisher',
                        '4': 'neurips2022nmmo.scripted.baselines.Carver',
                        '5': 'neurips2022nmmo.scripted.baselines.Range',
                        '6': 'neurips2022nmmo.scripted.baselines.Alchemist',
                        '7': 'neurips2022nmmo.scripted.baselines.Fisher',
                    }
                  }
