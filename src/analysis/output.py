# Writes trajectory data to output files
import pandas as pd
import datetime


class generate_excel_output:
    def __init__(self, spacecraft, force_model):
        self.spacecraft = spacecraft
        self.force_model = force_model
        self.meta = dict()
        self.trajectory = pd.DataFrame()

        self.times = self.spacecraft.integration_points
        self.orbital_parameters = self.spacecraft.orbital_parameters_track
        self.state_vectors = self.spacecraft.trajectory_track
        self.C3 = self.spacecraft.C3_track

        self.output_name = "trajectory"

        self.get_avaliable_data()
        self.write_body_trajectories()
        self.write_trajectory()

    def get_avaliable_data(self):
        if len(self.times) == 0:
            raise ValueError("No time vector available!")

        if len(self.orbital_parameters[0]) == 0:
            for k in range(len(self.times)):
                self.orbital_parameters[0].append(None)

                for i in range(1, 6):
                    self.orbital_parameters[i] = self.orbital_parameters[0]

        if len(self.C3) == 0:
            for k in range(len(self.times)):
                self.C3.append(None)

    def write_trajectory(self):
        self.trajectory["Time"] = self.times

        self.trajectory["X"] = self.state_vectors[0]
        self.trajectory["Y"] = self.state_vectors[1]
        self.trajectory["Z"] = self.state_vectors[2]
        self.trajectory["vX"] = self.state_vectors[3]
        self.trajectory["vY"] = self.state_vectors[4]
        self.trajectory["vZ"] = self.state_vectors[5]

        self.trajectory["SMA"] = self.orbital_parameters[0]
        self.trajectory["ECC"] = self.orbital_parameters[1]
        self.trajectory["INC"] = self.orbital_parameters[2]
        self.trajectory["RAAN"] = self.orbital_parameters[3]
        self.trajectory["APERI"] = self.orbital_parameters[4]
        self.trajectory["TAEPO"] = self.orbital_parameters[5]

        self.trajectory["C3"] = self.C3
        self.trajectory.to_excel(excel_writer="output/" + self.output_name + ".xlsx", sheet_name="trajectory")

    def write_body_trajectories(self):
        # Get the body trajectories
        writer = pd.ExcelWriter("output/" + "system_dynamics.xlsx", engine='xlsxwriter')
        trajectories = self.force_model.propagate_body_states(self.spacecraft.integration_points)
        for body in trajectories.keys():
            body_df = pd.DataFrame()
            body_df["X"] = trajectories[body][0]
            body_df["Y"] = trajectories[body][1]
            body_df["Z"] = trajectories[body][2]

            body_df.to_excel(excel_writer=writer, sheet_name=body)

        writer.close()
