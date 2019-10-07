﻿using System.Linq;
using System;
using UnityEngine;
using Random = UnityEngine.Random;
using MLAgents;
using PrefabInterface;

public class TrainingAgent : Agent, IPrefab
{
    public void RandomSize() { }
    public void SetColor(Vector3 color) { }
    public void SetSize(Vector3 scale) { }

    public virtual Vector3 GetPosition(Vector3 position,
                                        Vector3 boundingBox,
                                        float rangeX,
                                        float rangeZ)
    {
        float xBound = boundingBox.x;
        float zBound = boundingBox.z;
        float xOut = position.x < 0 ? Random.Range(xBound, rangeX - xBound)
                                    : Math.Max(0, Math.Min(position.x, rangeX));
        float yOut = Math.Max(position.y, 0) + transform.localScale.y / 2 + 0.01f;
        float zOut = position.z < 0 ? Random.Range(zBound, rangeZ - zBound)
                                    : Math.Max(0, Math.Min(position.z, rangeZ));

        return new Vector3(xOut, yOut, zOut);
    }

    public virtual Vector3 GetRotation(float rotationY)
    {
        return new Vector3(0,
                        rotationY < 0 ? Random.Range(0f, 360f) : rotationY,
                        0);
    }

    public float speed = 30f;
    public float rotationSpeed = 100f;
    public float rotationAngle = 0.25f;
    [HideInInspector]
    public int numberOfGoalsCollected = 0;

    private Rigidbody _rigidBody;
    private bool _isGrounded;
    private ContactPoint _lastContactPoint;
    private TrainingArea _area;
    private float _rewardPerStep;
    private Color[] _allBlackImage;
    private PlayerControls _playerScript;

    //private GameObject _goodgoal; // add by ali
    public GameObject goodgoal;
    private float reward_distance;

    //private TrainingArea _trainingArea;

    public override void InitializeAgent()
    { // this function is called when the last episode finished, either death or goal reached, 
        // and a new episode start, then initiallize a new agent to the new environment
        
        _area = GetComponentInParent<TrainingArea>();
        _rigidBody = GetComponent<Rigidbody>();
        _playerScript = GameObject.FindObjectOfType<PlayerControls>();
        // Negative reward to motivate the agent 
        _rewardPerStep = agentParameters.maxStep > 0 ? -1f / agentParameters.maxStep : 0; // can remove it 

        // Updating the reward 
        // float x,z;
        // x=Random.Range(0,20); //(min, max)
        // z=Random.Range(0,20);
        //_trainingArea.GetComponent<ArenaBuilders>()._goodgoal;
        Debug.Log("looking for good goal");
        goodgoal=GameObject.Find("GoodGoal");//return empty
        if(goodgoal==null)
        {
            Debug.Log("can not find goodgoal");
        }
        
        //Instantiate(_goodgoal, new Vector3(x,0,z),Quaternion.identity);
        //GameObject GoodGoal = Instantiate<GameObject>(GoodGoal)
        // diraction to goal  
        // _rewardPerStep = BallGoal.transform.position - transform.position
        // distance to goal
        
    }

    public override void CollectObservations()
    { // this function is called every time step
        Vector3 localVel = transform.InverseTransformDirection(_rigidBody.velocity);
        AddVectorObs(localVel); // only collect the velocity of the agent 

        AddVectorObs(Vector3.Distance(goodgoal.transform.position, this.transform.position)); // collect the distance information of the agent and goal
        AddVectorObs(this.transform.position); // collect the agent postion informaiton
        AddVectorObs(goodgoal.transform.position); // collect the goal ball postion information
        AddVectorObs(goodgoal.transform.InverseTransformDirection(goodgoal.GetComponent<Rigidbody>().velocity));
        // if you need to add more goals, like the multi goal or bad goal
        // you have to add their postion informaiton here too
    }

    public override void AgentAction(float[] vectorAction, string textAction)
    { // this functio is called every time step
        int actionForward = Mathf.FloorToInt(vectorAction[0]);
        int actionRotate = Mathf.FloorToInt(vectorAction[1]);

        moveAgent(actionForward, actionRotate);
        // here all the reward has to be update for each time step
        AddReward(_rewardPerStep); //punish time consuming 
        reward_distance=-(Vector3.Distance(goodgoal.transform.position, this.transform.position))/100.0f;
        AddReward(reward_distance); //punish the distance to goal
        // if you have other good goal or bad goal, have to add reward here too, AddReawrd(somthing );
    }

    private void moveAgent(int actionForward, int actionRotate)
    {
        Vector3 directionToGo = Vector3.zero;
        Vector3 rotateDirection = Vector3.zero;

        if (_isGrounded)
        {
            switch (actionForward)
            {
                case 1:
                // Go Forward
                    directionToGo = transform.forward * 1f;
                    break;
                case 2:
                // Go Backward
                    directionToGo = transform.forward * -1f;
                    break;
            }
        }
        switch (actionRotate)
        {
            case 1:
            // Turn Right
                rotateDirection = transform.up * 1f;
                break;
            case 2:
            // Turn Left 
                rotateDirection = transform.up * -1f;
                break;
        }

        transform.Rotate(rotateDirection, Time.fixedDeltaTime * rotationSpeed);
        _rigidBody.AddForce(directionToGo * speed * Time.fixedDeltaTime, ForceMode.VelocityChange);
    }

    public override void AgentReset()
    {
        _playerScript.prevScore = GetCumulativeReward();
        numberOfGoalsCollected = 0;
        _area.ResetArea();
        // add tiny negative reward 
        _rewardPerStep = agentParameters.maxStep > 0 ? -1f / agentParameters.maxStep : 0;
        _isGrounded = false;
    }


    void OnCollisionEnter(Collision collision)
    {
        foreach (ContactPoint contact in collision.contacts)
        {
            if (contact.normal.y > 0)
            {
                _isGrounded = true;
            }
        }
        _lastContactPoint = collision.contacts.Last();
    }

    void OnCollisionStay(Collision collision)
    {
        foreach (ContactPoint contact in collision.contacts)
        {
            if (contact.normal.y > 0)
            {
                _isGrounded = true;
            }
        }
        _lastContactPoint = collision.contacts.Last();
    }

    void OnCollisionExit(Collision collision)
    {
        if (_lastContactPoint.normal.y > 0)
        {
            _isGrounded = false;
        }
    }

    public void AgentDeath(float reward)
    {
        AddReward(reward);
        Done();
    }

    public void AddExtraReward(float rewardFactor)
    {
        AddReward(Math.Min(rewardFactor * _rewardPerStep,-0.00001f));
    }

    public override bool LightStatus()
    {
        return _area.UpdateLigthStatus(GetStepCount());
    }
}
