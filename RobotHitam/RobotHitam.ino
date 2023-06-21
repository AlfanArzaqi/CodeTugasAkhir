#include <ESP8266WiFi.h>
#include <IRremoteESP8266.h>
#include <IRsend.h>
#include <IRrecv.h>
#include <IRutils.h>
#include <WebSocketClient.h>

int port = 8876;
WiFiServer server(port);

const char *ssid = "Realme GT Master Edition";
const char *password = "12345678";

//const char *ssid = "ICOL";
//const char *password = "123sampai100";

// Mengatur IP Address ----------------------------------------------------
IPAddress local_IP(192, 168, 230, 118);
IPAddress gateway(192, 168, 1, 1);
IPAddress subnet(255, 255, 0, 0);

String data="";
String currentdata = "A";
int Mspeed = 100;
unsigned long prev_time;

const uint16_t kIrLed = 4;
IRsend irsend(kIrLed);

const uint16_t kRecvPin = 05;  
IRrecv irrecv(kRecvPin);
decode_results results;
int datashoot;

/* define L298N or L293D motor control pins */
int enA = 14;
int enB = 12;
int leftMotorForward = 2;     /* GPIO2(D4) -> IN3   */
int rightMotorForward = 13;   /* GPIO15(D8) -> IN1  */
int leftMotorBackward = 0;    /* GPIO0(D3) -> IN4   */
int rightMotorBackward = 15;  /* GPIO13(D7) -> IN2  */

void setup() {
  // put your setup code here, to run once:
  Serial.begin (115200);
  irsend.begin();
  irrecv.enableIRIn();
  Serial.println ();

  /* initialize motor control pins as output */
  pinMode (enA, OUTPUT);
  pinMode (enB, OUTPUT);
  pinMode(leftMotorForward, OUTPUT);
  pinMode(rightMotorForward, OUTPUT);
  pinMode(leftMotorBackward, OUTPUT);
  pinMode(rightMotorBackward, OUTPUT);

  // Configures static IP address
//  if (!WiFi.config(local_IP, gateway, subnet)) {
//    Serial.println("STA Failed to configure");
//  }
  
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  delay (1000); 
  Serial.println("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay (500);
    Serial.print(".");
    delay (500);
  }

  Serial.println("");
  Serial.print ("Connected to ");
  Serial.println (ssid);

  Serial.print ("IP Address: ");
  Serial.println(WiFi.localIP());
  server.begin ();
  Serial.print("Open Telnet and connect to IP: ");
  Serial.print (WiFi.localIP());
  Serial.print (" on port ");
  Serial.println (port);

  analogWrite(enA, Mspeed);
  analogWrite(enB, Mspeed);
}

void loop() {
  // put your main code here, to run repeatedly:
  WiFiClient client = server.available();

  if (client) {
    if (client.connected()){
      Serial.println("Client Connected");
    }
    while (client.connected()){
      while (Serial.available()>0){
      }
      while (client.available()>0){
        data = (char)client.read();
//        Mspeed = 100;
        Serial.println(data);

        if (irrecv.decode(&results)) {
          // print() & println() can't handle printing long longs. (uint64_t)
          datashoot = (results.value);
          serialPrintUint64(results.value, HEX);
          Serial.println(datashoot);
          Serial.println("");
          if (datashoot == 255){
            client.write(datashoot);
          }
          irrecv.resume();  // Receive the next value
        }
        

        /* If the incoming data is "forward", run the "MotorForward" function */
        if (data == "f") {
          if (currentdata != data) {
            Mspeed = 150;
          }
          else if (currentdata == data) {
            if (millis() - prev_time > 500){
              if (Mspeed < 250){
                Mspeed += 10;
              }
              else if (Mspeed > 250){
                Mspeed = 255;
              }
            }
          }

          prev_time = millis();
          currentdata = data;
          analogWrite(enA, Mspeed);
          analogWrite(enB, Mspeed);
          Serial.println(Mspeed);
          MotorForward();
        }
        /* If the incoming data is "backward", run the "MotorBackward" function */
        else if (data == "b") {
          if (currentdata != data) {
            Mspeed = 150;
          }
          else if (currentdata == data) {
            if (millis() - prev_time > 500){
              if (Mspeed < 250){
                Mspeed += 10;
              }
              else if (Mspeed > 250){
                Mspeed = 255;
              }
            }
          }

          prev_time = millis();
          currentdata = data;
          analogWrite(enA, Mspeed);
          analogWrite(enB, Mspeed);
          Serial.println(Mspeed);
          MotorBackward();
        }
        /* If the incoming data is "left", run the "TurnLeft" function */
        else if (data == "l") {
          if (currentdata != data) {
            Mspeed = 150;
          }
          else if (currentdata == data) {
            if (millis() - prev_time > 500){
              if (Mspeed < 250){
                Mspeed += 10;
              }
              else if (Mspeed > 250){
                Mspeed = 255;
              }
            }
          }

          prev_time = millis();
          currentdata = data;
          analogWrite(enA, Mspeed);
          analogWrite(enB, Mspeed);
          Serial.println(Mspeed);
          TurnLeft();
        }
        /* If the incoming data is "right", run the "TurnRight" function */
        else if (data == "r") {
          if (currentdata != data) {
            Mspeed = 150;
          }
          else if (currentdata == data) {
            if (millis() - prev_time > 500){
              if (Mspeed < 250){
                Mspeed += 10;
              }
              else if (Mspeed > 250){
                Mspeed = 255;
              }
            }
          }

          prev_time = millis();
          currentdata = data;
          analogWrite(enA, Mspeed);
          analogWrite(enB, Mspeed);
          Serial.println(Mspeed);
          TurnRight();
        }
        /* If the incoming data is "stop", run the "MotorStop" function */
        else if (data == "n" || data == "a") {
          MotorStop();
        }
        else if (data == "s") {
          MotorStop();
          irsend.sendNEC(0x00EE);
          Serial.println("Tembak");
        }
      }
      data = "";
    }
    client.stop();
    Serial.println ("Client disconnected");
    MotorStop();
  }

}

/********************************************* FORWARD *****************************************************/
void MotorForward(void)
{
  digitalWrite(leftMotorForward, HIGH);
  digitalWrite(rightMotorForward, HIGH);
  digitalWrite(rightMotorBackward, LOW);
  digitalWrite(leftMotorBackward, LOW);
}
 
/********************************************* BACKWARD *****************************************************/
void MotorBackward(void)
{
  digitalWrite(leftMotorBackward, HIGH);
  digitalWrite(rightMotorBackward, HIGH);
  digitalWrite(leftMotorForward, LOW);
  digitalWrite(rightMotorForward, LOW);
}
 
/********************************************* TURN LEFT *****************************************************/
void TurnLeft(void)
{
  digitalWrite(leftMotorForward, LOW);
  digitalWrite(rightMotorForward, HIGH);
  digitalWrite(rightMotorBackward, LOW);
  digitalWrite(leftMotorBackward, LOW);
}
 
/********************************************* TURN RIGHT *****************************************************/
void TurnRight(void)
{
  digitalWrite(leftMotorForward, HIGH);
  digitalWrite(rightMotorForward, LOW);
  digitalWrite(rightMotorBackward, LOW);
  digitalWrite(leftMotorBackward, LOW);
}
 
/********************************************* STOP *****************************************************/
void MotorStop(void)
{
  digitalWrite(leftMotorForward, LOW);
  digitalWrite(leftMotorBackward, LOW);
  digitalWrite(rightMotorForward, LOW);
  digitalWrite(rightMotorBackward, LOW);
}

void TambahKecepatan (String currentdata, String data)
{
  if (currentdata != data) {
    Mspeed = 100;
  }
  else if (currentdata == data) {
    if (millis() - prev_time > 500){
      if (Mspeed < 250){
        Mspeed += 10;
      }
      else if (Mspeed > 250){
        Mspeed = 255;
      }
      prev_time = millis();
    }
  }
  currentdata = data;
}
