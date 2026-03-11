from typing import List, Tuple, Optional, Union, Dict
from datetime import datetime
import math
import os
import yaml
import copy

from highway_env.road.road import Road, RoadNetwork, LaneIndex
from highway_env.road.lane import (
    StraightLane, CircularLane, SineLane, PolyLane, PolyLaneFixedWidth
)
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.vehicle.controller import MDPVehicle
from highway_env.vehicle.behavior import IDMVehicle
import numpy as np

from lsda.scenario.DBBridge import DBBridge
from lsda.scenario.envPlotter import ScePlotter


ACTIONS_ALL = {
    0: 'Turn-left',
    1: 'IDLE',
    2: 'Turn-right',
    3: 'Acceleration',
    4: 'Deceleration'
}

ACTIONS_DESCRIPTION = {
    0: 'Turn-left - change lane to the left of the current lane',
    1: 'IDLE - remain in the current lane with current speed',
    2: 'Turn-right - change lane to the right of the current lane',
    3: 'Acceleration - accelerate the vehicle',
    4: 'Deceleration - decelerate the vehicle'
}


class EnvScenario:
    def __init__(
            self, env: AbstractEnv, envType: str,
            seed: int, database: str = None
    ) -> None:
        self.env = env
        self.envType = envType

        self.ego: MDPVehicle = env.vehicle
        # The following four variables are used to determine whether a vehicle is within the ego's hazardous visibility range.
        self.theta1 = math.atan(3/17.5)
        self.theta2 = math.atan(2/2.5)
        self.radius1 = np.linalg.norm([3, 17.5])
        self.radius2 = np.linalg.norm([2, 2.5])

        self.road: Road = env.road
        self.network: RoadNetwork = self.road.network

        self.plotter = ScePlotter()
        if database:
            self.database = database
        else:
            self.database = datetime.strftime(
                datetime.now(), '%Y-%m-%d_%H-%M-%S'
            ) + '.db'

        if os.path.exists(self.database):
            os.remove(self.database)

        self.dbBridge = DBBridge(self.database, env)

        self.dbBridge.createTable()
        self.dbBridge.insertSimINFO(envType, seed)
        self.dbBridge.insertNetwork()
    # def getSurrendVehicles(self, vehicles_count: int) -> List[IDMVehicle]:
    #     return self.road.close_vehicles_to(
    #         self.ego, self.env.PERCEPTION_DISTANCE,
    #         count=vehicles_count-1, see_behind=True,
    #         sort='sorted'
    #     )
    def getSurrendVehicles(self, vehicles_count: int) -> List[IDMVehicle]:
        # Get vehicles within the perception range.
        vehicles = self.road.close_vehicles_to(
            self.ego, self.env.PERCEPTION_DISTANCE,
            count=vehicles_count-1, see_behind=True,
            sort='sorted'
        )
        
        return vehicles
        
    def plotSce(self, fileName: str) -> None:
        SVs = self.getSurrendVehicles(10)
        self.plotter.plotSce(self.network, SVs, self.ego, fileName)

    def getUnitVector(self, radian: float) -> Tuple[float]:
        return (
            math.cos(radian), math.sin(radian)
        )

    def isInJunction(self, vehicle: Union[IDMVehicle, MDPVehicle]) -> float:
        if self.envType == 'intersection-v1':
            x, y = vehicle.position
            # The junction is treated as roughly within [-20, 20] to ensure vehicles can detect information inside it.
            # In this situation, the vehicle should slow down in advance.
            if -20 <= x <= 20 and -20 <= y <= 20:
                return True
            else:
                return False
        else:
            return False

    def getLanePosition(self, vehicle: Union[IDMVehicle, MDPVehicle]) -> float:
        currentLaneIdx = vehicle.lane_index
        currentLane = self.network.get_lane(currentLaneIdx)
        if not isinstance(currentLane, StraightLane):
            raise ValueError(
                "The vehicle is in a junction, can't get lane position"
            )
        else:
            currentLane = self.network.get_lane(vehicle.lane_index)
            return np.linalg.norm(vehicle.position - currentLane.start)

    def availableActionsDescription(self) -> str:
        avaliableActionDescription = ''
        availableActions = self.env.get_available_actions()
        for action in availableActions:
            avaliableActionDescription += ACTIONS_DESCRIPTION[action] + ' Action_id: ' + str(
                action) + '\n'
        # if 1 in availableActions:
        #     avaliableActionDescription += 'You should check IDLE action as FIRST priority. '
        # if 0 in availableActions or 2 in availableActions:
        #     avaliableActionDescription += 'For change lane action, CAREFULLY CHECK the safety of vehicles on target lane. '
        # if 3 in availableActions:
        #     avaliableActionDescription += 'Consider acceleration action carefully. '
        # if 4 in availableActions:
        #     avaliableActionDescription += 'The deceleration action is LAST priority. '
        # avaliableActionDescription += '\n'
        return avaliableActionDescription

    def processNormalLane(self, lidx: LaneIndex) -> str:
        sideLanes = self.network.all_side_lanes(lidx)
        numLanes = len(sideLanes)
        if numLanes == 1:
            description = "You are driving on a road with only one lane, you can't change lane. "
        else:
            egoLaneRank = lidx[2]
            if egoLaneRank == 0:
                description = f"You are driving on a road with {numLanes} lanes, and you are currently driving in the leftmost lane. "
            elif egoLaneRank == numLanes - 1:
                description = f"You are driving on a road with {numLanes} lanes, and you are currently driving in the rightmost lane. "
            else:
                laneRankDict = {
                    1: 'second',
                    2: 'third',
                    3: 'fourth',
                    4: 'fifth',
                    5: 'sixth',
                    6: 'seventh',
                    7: 'eighth',
                    8: 'ninth',
                    9: 'tenth'
                }
                description = f"You are driving on a road with {numLanes} lanes, and you are currently driving in the {laneRankDict[egoLaneRank]} lane from the left. "

        description += f"Your current position is `({self.ego.position[0]:.2f}, {self.ego.position[1]:.2f})`, speed is {self.ego.speed:.2f} m/s, acceleration is {self.ego.action['acceleration']:.2f} m/s^2, and lane position is {self.getLanePosition(self.ego):.2f} m.\n"
        return description

    def getSVRelativeState(self, sv: IDMVehicle) -> str:
        # CAUTION: pygame's y-axis is inverted (positive y points downward).
        # Therefore, in highway-v0, a "change lane left" action visually moves right.
        # For relative left/right position, it is more reliable to use lane indices than vectors.
        # Vectors are only used here to decide whether the vehicle is ahead of or behind the ego.
        relativePosition = sv.position - self.ego.position
        egoUnitVector = self.getUnitVector(self.ego.heading)
        cosineValue = sum(
            [x*y for x, y in zip(relativePosition, egoUnitVector)]
        )
        if cosineValue >= 0:
            return 'is ahead of you'
        else:
            return 'is behind of you'

    def getVehDis(self, veh: IDMVehicle):
        posA = self.ego.position
        posB = veh.position
        distance = np.linalg.norm(posA - posB)
        return distance

    def getClosestSV(self, SVs: List[IDMVehicle]):
        if SVs:
            closestIdex = -1
            closestDis = 99999999
            for i, sv in enumerate(SVs):
                dis = self.getVehDis(sv)
                if dis < closestDis:
                    closestDis = dis
                    closestIdex = i
            return SVs[closestIdex]
        else:
            return None

    def processSingleLaneSVs(self, SingleLaneSVs: List[IDMVehicle]):
        # Return the closest vehicle ahead and behind on the current lane; None if absent.
        if SingleLaneSVs:
            aheadSVs = []
            behindSVs = []
            for sv in SingleLaneSVs:
                RSStr = self.getSVRelativeState(sv)
                if RSStr == 'is ahead of you':
                    aheadSVs.append(sv)
                else:
                    behindSVs.append(sv)
            aheadClosestOne = self.getClosestSV(aheadSVs)
            behindClosestOne = self.getClosestSV(behindSVs)
            return aheadClosestOne, behindClosestOne
        else:
            return None, None

    def processSVsNormalLane(
            self, SVs: List[IDMVehicle], currentLaneIndex: LaneIndex
    ):
        # The description can contain too many vehicles; keep only the few closest to the ego vehicle.
        classifiedSVs: Dict[str, List[IDMVehicle]] = {
            'current lane': [],
            'left lane': [],
            'right lane': [],
            'target lane': []
        }
        sideLanes = self.network.all_side_lanes(currentLaneIndex)
        nextLane = self.network.next_lane(
            currentLaneIndex, self.ego.route, self.ego.position
        )
        for sv in SVs:
            lidx = sv.lane_index
            if lidx in sideLanes:
                if lidx == currentLaneIndex:
                    classifiedSVs['current lane'].append(sv)
                else:
                    laneRelative = lidx[2] - currentLaneIndex[2]
                    if laneRelative == 1:
                        classifiedSVs['right lane'].append(sv)
                    elif laneRelative == -1:
                        classifiedSVs['left lane'].append(sv)
                    else:
                        continue
            elif lidx == nextLane:
                classifiedSVs['target lane'].append(sv)
            else:
                continue

        validVehicles: List[IDMVehicle] = []
        existVehicles: Dict[str, bool] = {}
        for k, v in classifiedSVs.items():
            if v:
                existVehicles[k] = True
            else:
                existVehicles[k] = False
            ahead, behind = self.processSingleLaneSVs(v)
            if ahead:
                validVehicles.append(ahead)
            if behind:
                validVehicles.append(behind)

        return validVehicles, existVehicles

    def describeSVNormalLane(self, currentLaneIndex: LaneIndex) -> str:
        # When the ego is on a StraightLane, lane information matters and must be processed.
        # First, determine whether a vehicle is on the same road as the ego:
        #   - If on the same road, determine which lane it is on (relative to the ego).
        #   - If not on the same road, check whether it is on the next_lane:
        #       - If not on nextLane, ignore it.
        #       - If on nextLane, record its relative motion w.r.t. the ego.
        sideLanes = self.network.all_side_lanes(currentLaneIndex)
        nextLane = self.network.next_lane(
            currentLaneIndex, self.ego.route, self.ego.position
        )
        surroundVehicles = self.getSurrendVehicles(10)
        validVehicles, existVehicles = self.processSVsNormalLane(
            surroundVehicles, currentLaneIndex
        )
        if not surroundVehicles:
            SVDescription = "There are no other vehicles driving near you, so you can drive completely according to your own ideas.\n"
            return SVDescription
        else:
            SVDescription = ''
            for sv in surroundVehicles:
                lidx = sv.lane_index
                if lidx in sideLanes:
                    # Vehicle is driving on the same road as the ego vehicle.
                    if lidx == currentLaneIndex:
                        # Vehicle is driving on the same lane as the ego vehicle.
                        if sv in validVehicles:
                            SVDescription += f"- Vehicle `{id(sv) % 1000}` is driving on the same lane as you and {self.getSVRelativeState(sv)}. "
                        else:
                            continue
                    else:
                        laneRelative = lidx[2] - currentLaneIndex[2]
                        if laneRelative == 1:
                            # laneRelative = 1 means the vehicle is driving in the lane to the right of the ego vehicle.
                            if sv in validVehicles:
                                SVDescription += f"- Vehicle `{id(sv) % 1000}` is driving on the lane to your right and {self.getSVRelativeState(sv)}. "
                            else:
                                continue
                        elif laneRelative == -1:
                            # laneRelative = -1 means the vehicle is driving in the lane to the left of the ego vehicle.
                            if sv in validVehicles:
                                SVDescription += f"- Vehicle `{id(sv) % 1000}` is driving on the lane to your left and {self.getSVRelativeState(sv)}. "
                            else:
                                continue
                        else:
                            # Other laneRelative values indicate lanes further away and can be ignored.
                            continue
                elif lidx == nextLane:
                    # Vehicle is driving on the ego's nextLane.
                    if sv in validVehicles:
                        SVDescription += f"- Vehicle `{id(sv) % 1000}` is driving on your target lane and {self.getSVRelativeState(sv)}. "
                    else:
                        continue
                else:
                    continue
                if self.envType == 'intersection-v1':
                    SVDescription += f"The position of it is `({sv.position[0]:.2f}, {sv.position[1]:.2f})`, speed is {sv.speed:.2f} m/s, acceleration is {sv.action['acceleration']:.2f} m/s^2.\n"
                else:
                    SVDescription += f"The position of it is `({sv.position[0]:.2f}, {sv.position[1]:.2f})`, speed is {sv.speed:.2f} m/s, acceleration is {sv.action['acceleration']:.2f} m/s^2, and lane position is {self.getLanePosition(sv):.2f} m.\n"
            if SVDescription:
                descriptionPrefix = "There are other vehicles driving around you, and below is their basic information:\n"
                return descriptionPrefix + SVDescription
            else:
                SVDescription = 'There are no other vehicles driving near you, so you can drive completely according to your own ideas.\n'
                return SVDescription

    def isInDangerousArea(self, sv: IDMVehicle) -> bool:
        relativeVector = sv.position - self.ego.position
        distance = np.linalg.norm(relativeVector)
        egoUnitVector = self.getUnitVector(self.ego.heading)
        relativeUnitVector = relativeVector / distance
        alpha = np.arccos(
            np.clip(np.dot(egoUnitVector, relativeUnitVector), -1, 1)
        )
        if alpha <= self.theta1:
            if distance <= self.radius1:
                return True
            else:
                return False
        elif self.theta1 < alpha <= self.theta2:
            if distance <= self.radius2:
                return True
            else:
                return False
        else:
            return False

    def describeSVJunctionLane(self, currentLaneIndex: LaneIndex) -> str:
        # When the ego is inside a junction, lane information becomes less important; only relative position matters.
        # However, we still need to determine the position of all junction lanes relative to the ego.
        nextLane = self.network.next_lane(
            currentLaneIndex, self.ego.route, self.ego.position
        )
        surroundVehicles = self.getSurrendVehicles(6)
        if not surroundVehicles:
            SVDescription = "There are no other vehicles driving near you, so you can drive completely according to your own ideas.\n"
            return SVDescription
        else:
            SVDescription = ''
            for sv in surroundVehicles:
                lidx = sv.lane_index
                if self.isInJunction(sv):
                    collisionPoint = self.getCollisionPoint(sv)
                    if collisionPoint:
                        SVDescription += f"- Vehicle `{id(sv) % 1000}` is also in the junction and {self.getSVRelativeState(sv)}. The position of it is `({sv.position[0]:.2f}, {sv.position[1]:.2f})`, speed is {sv.speed:.2f} m/s, and acceleration is {sv.action['acceleration']:.2f} m/s^2. The potential collision point is `({collisionPoint[0]:.2f}, {collisionPoint[1]:.2f})`.\n"
                    else:
                        SVDescription += f"- Vehicle `{id(sv) % 1000}` is also in the junction and {self.getSVRelativeState(sv)}. The position of it is `({sv.position[0]:.2f}, {sv.position[1]:.2f})`, speed is {sv.speed:.2f} m/s, and acceleration is {sv.action['acceleration']:.2f} m/s^2. You two are no potential collision.\n"
                elif lidx == nextLane:
                    collisionPoint = self.getCollisionPoint(sv)
                    if collisionPoint:
                        SVDescription += f"- Vehicle `{id(sv) % 1000}` is driving on your target lane and {self.getSVRelativeState(sv)}. The position of it is `({sv.position[0]:.2f}, {sv.position[1]:.2f})`, speed is {sv.speed:.2f} m/s, and acceleration is {sv.action['acceleration']:.2f} m/s^2. The potential collision point is `({collisionPoint[0]:.2f}, {collisionPoint[1]:.2f})`.\n"
                    else:
                        SVDescription += f"- Vehicle `{id(sv) % 1000}` is driving on your target lane and {self.getSVRelativeState(sv)}. The position of it is `({sv.position[0]:.2f}, {sv.position[1]:.2f})`, speed is {sv.speed:.2f} m/s, and acceleration is {sv.action['acceleration']:.2f} m/s^2. You two are no potential collision.\n"
                if self.isInDangerousArea(sv):
                    print(f"Vehicle {id(sv) % 1000} is in dangerous area.")
                    SVDescription += f"- Vehicle `{id(sv) % 1000}` is also in the junction and {self.getSVRelativeState(sv)}. The position of it is `({sv.position[0]:.2f}, {sv.position[1]:.2f})`, speed is {sv.speed:.2f} m/s, and acceleration is {sv.action['acceleration']:.2f} m/s^2. This car is within your field of vision, and you need to pay attention to its status when making decisions.\n"
                else:
                    continue
            if SVDescription:
                descriptionPrefix = "There are other vehicles driving around you, and below is their basic information:\n"
                return descriptionPrefix + SVDescription
            else:
                'There are no other vehicles driving near you, so you can drive completely according to your own ideas.\n'
                return SVDescription

    # def describe(self, decisionFrame: int) -> str:
    #     surroundVehicles = self.getSurrendVehicles(10)
    #     self.dbBridge.insertVehicle(decisionFrame, surroundVehicles)
    #     currentLaneIndex: LaneIndex = self.ego.lane_index
    #     if self.isInJunction(self.ego):
    #         roadCondition = "You are driving in an intersection, you can't change lane. "
    #         roadCondition += f"Your current position is `({self.ego.position[0]:.2f}, {self.ego.position[1]:.2f})`, speed is {self.ego.speed:.2f} m/s, and acceleration is {self.ego.action['acceleration']:.2f} m/s^2.\n"
    #         SVDescription = self.describeSVJunctionLane(currentLaneIndex)
    #     else:
    #         roadCondition = self.processNormalLane(currentLaneIndex)
    #         SVDescription = self.describeSVNormalLane(currentLaneIndex)

    #     return roadCondition + SVDescription
    def describe(self, decisionFrame: int) -> str:
        # Continue describing the scenario.
        surroundVehicles = self.getSurrendVehicles(10)
        self.dbBridge.insertVehicle(decisionFrame, surroundVehicles)
        currentLaneIndex: LaneIndex = self.ego.lane_index
        if self.isInJunction(self.ego):
            roadCondition = "You are driving in an intersection, you can't change lane. "
            roadCondition += f"Your current position is `({self.ego.position[0]:.2f}, {self.ego.position[1]:.2f})`, speed is {self.ego.speed:.2f} m/s, and acceleration is {self.ego.action['acceleration']:.2f} m/s^2.\n"
            SVDescription = self.describeSVJunctionLane(currentLaneIndex)
        else:
            roadCondition = self.processNormalLane(currentLaneIndex)
            SVDescription = self.describeSVNormalLane(currentLaneIndex)
        
        return roadCondition + SVDescription
    
    def promptsCommit(
        self, decisionFrame: int, vectorID: str, done: bool,
        description: str, fewshots: str, thoughtsAndAction: str
    ):
        self.dbBridge.insertPrompts(
            decisionFrame, vectorID, done, description,
            fewshots, thoughtsAndAction
        )

    def sync_environment(self, env: AbstractEnv) -> None:
        """Synchronize internal references to match the latest environment instance.

        When the underlying environment is further wrapped (e.g., video recording, visualization)
        or reset, call this method to ensure that scenario descriptions and database writes
        are based on the current ego vehicle and road.
        """
        self.env = env
        self.ego = env.vehicle
        self.road = env.road
        self.network = self.road.network

        # Synchronize references in the database bridge to avoid writing stale state.
        if hasattr(self, 'dbBridge'):
            self.dbBridge.env = env
            self.dbBridge.ego = self.ego
            self.dbBridge.network = self.network
