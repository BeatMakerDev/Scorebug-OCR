#define MyAppName "Scorebug OCR"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "WBHS9"
#define MyAppExeName "ScorebugOCR.exe"

[Setup]
AppId={{12345678-ABCD-4321-DCBA-876543210000}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
OutputDir=.
OutputBaseFilename=Scorebug_OCR_Installer
Compression=lzma
SolidCompression=yes
WizardStyle=modern
SetupIconFile=assets\scorebug.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
ArchitecturesInstallIn64BitMode=x64

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
; Main application EXE
Source: "dist\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion


[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop icon"; Flags: unchecked

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent
