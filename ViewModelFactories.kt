package com.example.werkzeugerkennung

import android.Manifest
import android.content.Context
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.compose.setContent
import androidx.compose.foundation.Image
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Checkbox
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Divider
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import androidx.lifecycle.viewmodel.compose.viewModel
import java.io.InputStream
import kotlinx.coroutines.launch
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import androidx.core.content.ContextCompat

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme {
                val context = LocalContext.current
                var backendUrl by remember { mutableStateOf(BackendSettings.loadBaseUrl(context)) }
                var backendApiKey by remember { mutableStateOf(BackendSettings.loadApiKey(context)) }
                var repository by remember { mutableStateOf<BackendRepository?>(null) }
                var initError by remember { mutableStateOf<String?>(null) }
                var backendKey by remember { mutableStateOf("init") }

                LaunchedEffect(backendUrl, backendApiKey) {
                    try {
                        val repo = withContext(Dispatchers.IO) {
                            createRepository(context.applicationContext, backendUrl, backendApiKey)
                        }
                        repository = repo
                        backendKey = "${backendUrl}|${backendApiKey}"
                    } catch (e: Exception) {
                        initError = e.message ?: "Backend initialization failed."
                    }
                }

                if (repository == null) {
                    BackendLoadingScreen(message = initError)
                } else {
                    AppRoot(
                        repository = repository!!,
                        backendUrl = backendUrl,
                        backendApiKey = backendApiKey,
                        backendKey = backendKey,
                        onBackendSettingsChange = { rawUrl, rawKey ->
                            backendUrl = BackendSettings.saveBaseUrl(context, rawUrl)
                            backendApiKey = BackendSettings.saveApiKey(context, rawKey)
                        }
                    )
                }
            }
        }
    }
}

private sealed class Screen {
    data object Session : Screen()
    data class Capture(
        val objectId: String,
        val objectName: String,
        val forceComplete: Boolean = false
    ) : Screen()
    data object TrainingSelection : Screen()
    data object TrainingLists : Screen()
    data object ThresholdSettings : Screen()
    data object Inference : Screen()
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun AppRoot(
    repository: BackendRepository,
    backendUrl: String,
    backendApiKey: String,
    backendKey: String,
    onBackendSettingsChange: (String, String) -> Unit
) {
    // The app is a simple flow:
    // 1) Select 3 environments (one-by-one confirmation)
    // 2) Create objects
    // 3) Capture required shots per object
    // 4) Build training sets
    val sessionViewModel: SessionViewModel = viewModel(
        key = "session-$backendKey",
        factory = SessionViewModelFactory(repository)
    )
    val trainingViewModel: TrainingViewModel = viewModel(
        key = "training-$backendKey",
        factory = TrainingViewModelFactory(repository)
    )
    val inferenceViewModel: InferenceViewModel = viewModel(
        key = "inference-$backendKey",
        factory = InferenceViewModelFactory(repository)
    )
    var screen by remember { mutableStateOf<Screen>(Screen.Session) }
    val snackbarHostState = remember { SnackbarHostState() }
    val scope = rememberCoroutineScope()

    LaunchedEffect(backendKey) {
        sessionViewModel.loadSession()
    }

    LaunchedEffect(backendKey) {
        screen = Screen.Session
    }

    val sessionError by sessionViewModel.errorMessage.collectAsState()
    LaunchedEffect(sessionError) {
        sessionError?.let {
            scope.launch { snackbarHostState.showSnackbar(it) }
            sessionViewModel.clearError()
        }
    }

    val trainingError by trainingViewModel.errorMessage.collectAsState()
    LaunchedEffect(trainingError) {
        trainingError?.let {
            scope.launch { snackbarHostState.showSnackbar(it) }
            trainingViewModel.clearError()
        }
    }

    Scaffold(
        topBar = { TopAppBar(title = { Text("Werkzeugerkennung") }) },
        snackbarHost = { SnackbarHost(snackbarHostState) }
    ) { paddingValues ->
        Box(modifier = Modifier.padding(paddingValues)) {
            when (val current = screen) {
                Screen.Session -> SessionScreen(
                    sessionViewModel = sessionViewModel,
                    trainingViewModel = trainingViewModel,
                    backendUrl = backendUrl,
                    backendApiKey = backendApiKey,
                    onBackendSettingsChange = onBackendSettingsChange,
                    onCapture = { objectId, objectName ->
                        screen = Screen.Capture(objectId, objectName)
                    },
                    onInference = { screen = Screen.Inference },
                    onTrainingSelection = { screen = Screen.TrainingSelection },
                    onTrainingLists = { screen = Screen.TrainingLists },
                    onThresholdSettings = { screen = Screen.ThresholdSettings }
                )

                is Screen.Capture -> CaptureScreen(
                    repository = repository,
                    objectId = current.objectId,
                    objectName = current.objectName,
                    envSpec = sessionViewModel.envSpec.collectAsState().value,
                    backendKey = backendKey,
                    forceComplete = current.forceComplete,
                    onBack = {
                        screen = Screen.Session
                        sessionViewModel.refreshObjects()
                    }
                )

                Screen.TrainingSelection -> TrainingSelectionScreen(
                    envSpec = sessionViewModel.envSpec.collectAsState().value,
                    trainingViewModel = trainingViewModel,
                    onBack = { screen = Screen.Session }
                )

                Screen.TrainingLists -> TrainingListsScreen(
                    trainingViewModel = trainingViewModel,
                    onBack = { screen = Screen.Session }
                )

                Screen.ThresholdSettings -> ThresholdSettingsScreen(
                    sessionViewModel = sessionViewModel,
                    backendUrl = backendUrl,
                    onBack = { screen = Screen.Session }
                )

                Screen.Inference -> InferenceScreen(
                    inferenceViewModel = inferenceViewModel,
                    onBack = { screen = Screen.Session },
                    onStartSequence = { objectId, objectName ->
                        screen = Screen.Capture(objectId, objectName, true)
                    }
                )
            }
        }
    }
}

@Composable
private fun BackendLoadingScreen(message: String?) {
    Column(
        modifier = Modifier.fillMaxSize().padding(16.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        CircularProgressIndicator()
        Spacer(modifier = Modifier.height(12.dp))
        Text("Starting backend...")
        if (!message.isNullOrBlank()) {
            Spacer(modifier = Modifier.height(8.dp))
            Text(message, color = MaterialTheme.colorScheme.error)
        }
    }
}

@Composable
private fun BlockingLoadingDialog(message: String) {
    Dialog(
        onDismissRequest = {},
        properties = DialogProperties(
            dismissOnBackPress = false,
            dismissOnClickOutside = false
        )
    ) {
        Surface(shape = MaterialTheme.shapes.medium, tonalElevation = 6.dp) {
            Row(
                modifier = Modifier.padding(24.dp),
                horizontalArrangement = Arrangement.spacedBy(12.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                CircularProgressIndicator(
                    modifier = Modifier.size(20.dp),
                    strokeWidth = 2.dp
                )
                Text(message)
            }
        }
    }
}

@Composable
private fun BackendSettingsSection(
    backendUrl: String,
    backendApiKey: String,
    onBackendSettingsChange: (String, String) -> Unit
) {
    var inputUrl by remember(backendUrl) { mutableStateOf(backendUrl) }
    var inputApiKey by remember(backendApiKey) { mutableStateOf(backendApiKey) }
    Column {
        Text("Backend", style = MaterialTheme.typography.titleMedium)
        Spacer(modifier = Modifier.height(6.dp))
        OutlinedTextField(
            value = inputUrl,
            onValueChange = { inputUrl = it },
            label = { Text("Backend base URL") },
            placeholder = { Text(BackendSettings.DEFAULT_BASE_URL) },
            modifier = Modifier.fillMaxWidth()
        )
        Spacer(modifier = Modifier.height(6.dp))
        OutlinedTextField(
            value = inputApiKey,
            onValueChange = { inputApiKey = it },
            label = { Text("API key (optional)") },
            modifier = Modifier.fillMaxWidth()
        )
        Spacer(modifier = Modifier.height(6.dp))
        Text("Backend URL is required (blank resets to default).")
        Text("API key is sent as X-API-Key for requests.")
        Spacer(modifier = Modifier.height(6.dp))
        Button(onClick = { onBackendSettingsChange(inputUrl, inputApiKey) }) {
            Text("Apply")
        }
    }
}

@Composable
private fun SessionScreen(
    sessionViewModel: SessionViewModel,
    trainingViewModel: TrainingViewModel,
    backendUrl: String,
    backendApiKey: String,
    onBackendSettingsChange: (String, String) -> Unit,
    onCapture: (String, String) -> Unit,
    onInference: () -> Unit,
    onTrainingSelection: () -> Unit,
    onTrainingLists: () -> Unit,
    onThresholdSettings: () -> Unit
) {
    val objects by sessionViewModel.objects.collectAsState()
    val envSpec by sessionViewModel.envSpec.collectAsState()
    val isLoading by sessionViewModel.isLoading.collectAsState()
    val softResetStatus by sessionViewModel.softResetStatus.collectAsState()
    val selectionState by trainingViewModel.selectionState.collectAsState()
    var showCreateDialog by remember { mutableStateOf(false) }
    var showResetDialog by remember { mutableStateOf(false) }
    var showSoftResetDialog by remember { mutableStateOf(false) }
    val selectionRemaining = selectionState.remainingToSelect
    val selectionComplete = selectionRemaining == 0

    LaunchedEffect(Unit) {
        trainingViewModel.loadSelection()
        sessionViewModel.loadSoftResetStatus()
    }
    LaunchedEffect(objects, selectionState.remainingToSelect) {
        sessionViewModel.loadSoftResetStatus()
    }

    Column(modifier = Modifier.fillMaxSize().padding(16.dp)) {
        Text("Session / Initialization", style = MaterialTheme.typography.titleLarge)
        Spacer(modifier = Modifier.height(8.dp))
        Text("Settings", style = MaterialTheme.typography.titleMedium)
        Spacer(modifier = Modifier.height(6.dp))
        BackendSettingsSection(
            backendUrl = backendUrl,
            backendApiKey = backendApiKey,
            onBackendSettingsChange = onBackendSettingsChange
        )
        Spacer(modifier = Modifier.height(8.dp))
        Button(onClick = onThresholdSettings) {
            Text("Threshold Settings")
        }
        Spacer(modifier = Modifier.height(12.dp))
        Divider()
        Spacer(modifier = Modifier.height(12.dp))
        Text("Environment definitions loaded: ${envSpec.size}")
        if (isLoading) {
            Spacer(modifier = Modifier.height(8.dp))
            Text("Loading...", style = MaterialTheme.typography.bodyMedium)
        }
        Spacer(modifier = Modifier.height(12.dp))
        Text("Environment selection remaining: $selectionRemaining")
        Spacer(modifier = Modifier.height(8.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(
                onClick = { showCreateDialog = true },
                enabled = selectionComplete
            ) {
                Text("Create Object")
            }
            OutlinedButton(onClick = sessionViewModel::refreshObjects) {
                Text("Refresh")
            }
        }
        if (!selectionComplete) {
            Spacer(modifier = Modifier.height(6.dp))
            Text("Select 3 environments before creating objects.")
        }
        Spacer(modifier = Modifier.height(12.dp))
        Text("Objects", style = MaterialTheme.typography.titleMedium)
        Divider(modifier = Modifier.padding(vertical = 6.dp))
        LazyColumn(modifier = Modifier.weight(1f, fill = true)) {
            items(objects) { record ->
                ObjectRow(record = record, onCapture = onCapture)
            }
        }
        Spacer(modifier = Modifier.height(12.dp))
        Text("Training", style = MaterialTheme.typography.titleMedium)
        Spacer(modifier = Modifier.height(4.dp))
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Button(
                onClick = onTrainingSelection,
                enabled = true,
                modifier = Modifier.weight(1f)
            ) {
                Text("Select Environments")
            }
            OutlinedButton(
                onClick = onTrainingLists,
                enabled = objects.isNotEmpty(),
                modifier = Modifier.weight(1f)
            ) {
                Text("Training Lists")
            }
        }
        Spacer(modifier = Modifier.height(6.dp))
        Button(
            onClick = onInference
        ) {
            Text("Start Inference")
        }
        Spacer(modifier = Modifier.height(12.dp))
        val canSoftReset = softResetStatus?.canSoftReset == true
        Button(
            onClick = { showSoftResetDialog = true },
            enabled = canSoftReset
        ) {
            Text("Soft Reset")
        }
        if (!canSoftReset) {
            val missing = softResetStatus?.missing ?: emptyList()
            val reasons = buildList {
                for (item in missing) {
                    when (item) {
                        "envs_selected" -> add("select 3 environments")
                        "objects_created" -> add("create baseline objects")
                        "training_lists_complete" -> add("finish 5-shot training lists")
                        "inference_started" -> add("run in-the-wild inference once")
                        "thresholds_locked" -> add("lock thresholds")
                    }
                }
            }
            if (reasons.isNotEmpty()) {
                Spacer(modifier = Modifier.height(6.dp))
                Text("Soft reset locked until you: ${reasons.joinToString(", ")}.")
            }
        }
        Spacer(modifier = Modifier.height(12.dp))
        OutlinedButton(onClick = { showResetDialog = true }) {
            Text("Start Over (Delete All Objects)", color = MaterialTheme.colorScheme.error)
        }
    }

    if (showCreateDialog) {
        CreateObjectDialog(
            onDismiss = { showCreateDialog = false },
            onCreate = {
                sessionViewModel.createObject(it)
                showCreateDialog = false
            }
        )
    }

    if (showResetDialog) {
        AlertDialog(
            onDismissRequest = { showResetDialog = false },
            title = { Text("Start over?") },
            text = {
                Text("This will delete all objects and clear training selection. This cannot be undone.")
            },
            confirmButton = {
                Button(
                    onClick = {
                        sessionViewModel.resetAll()
                        showResetDialog = false
                    }
                ) {
                    Text("Delete")
                }
            },
            dismissButton = {
                OutlinedButton(onClick = { showResetDialog = false }) {
                    Text("Cancel")
                }
            }
        )
    }

    if (showSoftResetDialog) {
        AlertDialog(
            onDismissRequest = { showSoftResetDialog = false },
            title = { Text("Soft reset?") },
            text = {
                Text(
                    "This removes PostTraining shots and deletes objects created in the wild. " +
                        "Baseline objects and their 5-shot training lists remain."
                )
            },
            confirmButton = {
                Button(
                    onClick = {
                        sessionViewModel.softReset()
                        showSoftResetDialog = false
                    }
                ) {
                    Text("Soft Reset")
                }
            },
            dismissButton = {
                OutlinedButton(onClick = { showSoftResetDialog = false }) {
                    Text("Cancel")
                }
            }
        )
    }

    if (isLoading) {
        BlockingLoadingDialog("Updating session...")
    }
}

@Composable
private fun ObjectRow(record: ObjectRecord, onCapture: (String, String) -> Unit) {
    Column(modifier = Modifier.fillMaxWidth().padding(vertical = 6.dp)) {
        Text(record.objectName.ifBlank { record.objectId }, fontWeight = FontWeight.SemiBold)
        Text("Instance: ${record.instanceId}")
        Text("Completed: ${record.completed}")
        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.End) {
            OutlinedButton(onClick = { onCapture(record.objectId, record.objectName) }) {
                Text("Continue Capture")
            }
        }
        Divider(modifier = Modifier.padding(top = 8.dp))
    }
}

@Composable
private fun CreateObjectDialog(onDismiss: () -> Unit, onCreate: (String) -> Unit) {
    var name by remember { mutableStateOf("") }
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Create Object") },
        text = {
            Column {
                Text("Enter a display name for the object.")
                Spacer(modifier = Modifier.height(8.dp))
                OutlinedTextField(
                    value = name,
                    onValueChange = { name = it },
                    label = { Text("Object name") },
                    modifier = Modifier.fillMaxWidth()
                )
            }
        },
        confirmButton = {
            Button(onClick = { onCreate(name) }, enabled = name.isNotBlank()) {
                Text("Create")
            }
        },
        dismissButton = {
            OutlinedButton(onClick = onDismiss) { Text("Cancel") }
        }
    )
}

@Composable
private fun CaptureScreen(
    repository: BackendRepository,
    objectId: String,
    objectName: String,
    envSpec: List<EnvSpecItem>,
    backendKey: String,
    forceComplete: Boolean,
    onBack: () -> Unit
) {
    // Shot capture is driven by backend state:
    // the backend tells us the exact required env_code at each step.
    val viewModel: ShotViewModel = viewModel(
        key = "$backendKey-$objectId",
        factory = ShotViewModelFactory(repository, objectId)
    )
    val nextShot by viewModel.nextShot.collectAsState()
    val errorMessage by viewModel.errorMessage.collectAsState()
    val isSubmitting by viewModel.isSubmitting.collectAsState()
    val context = LocalContext.current
    var previewBytes by remember { mutableStateOf<ByteArray?>(null) }
    var captureRequested by remember { mutableStateOf(false) }
    var pendingCapture by remember { mutableStateOf(false) }
    val galleryLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri ->
        previewBytes = uri?.let { context.readBytesFromUri(it) }
    }
    val cameraPermissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted && pendingCapture) {
            captureRequested = true
        }
        pendingCapture = false
    }
    LaunchedEffect(objectId) {
        viewModel.loadNextShot()
    }

    val envDisplayName = envSpec.firstOrNull { it.envCode == nextShot?.envCode }
        ?.let { it.friendlyName.ifBlank { it.envName }.ifBlank { it.envCode } }
    val objectCompleted = nextShot?.objectCompleted == true
    if (forceComplete && !objectCompleted) {
        BackHandler(enabled = true) {}
    }

    Column(modifier = Modifier.fillMaxSize().padding(16.dp)) {
        Text("Shot Collection", style = MaterialTheme.typography.titleLarge)
        Spacer(modifier = Modifier.height(8.dp))
        Text("Object: ${objectName.ifBlank { objectId }}")
        Spacer(modifier = Modifier.height(12.dp))
        if (objectCompleted) {
            Text("Object capture complete.")
            Spacer(modifier = Modifier.height(12.dp))
            Button(onClick = onBack) { Text("Back to Objects") }
        } else {
            CameraCapture(
                captureTrigger = captureRequested,
                onImageCaptured = { bytes ->
                    previewBytes = bytes
                    captureRequested = false
                },
                onError = {
                    captureRequested = false
                }
            )
            Spacer(modifier = Modifier.height(8.dp))
            Surface(
                modifier = Modifier.fillMaxWidth(),
                tonalElevation = 2.dp,
                shape = MaterialTheme.shapes.medium
            ) {
                Column(modifier = Modifier.padding(12.dp)) {
                    if (nextShot == null) {
                        Text("Loading next required shot...")
                    } else {
                        Text("Required environment: ${nextShot?.envCode} (${envDisplayName ?: "Unknown"})")
                        Text("Remaining shots for env: ${nextShot?.remainingNeededForEnv}")
                    }
                }
            }
            Spacer(modifier = Modifier.height(8.dp))
            if (errorMessage != null) {
                Text(
                    text = errorMessage ?: "",
                    color = MaterialTheme.colorScheme.error,
                    style = MaterialTheme.typography.bodyMedium
                )
                Spacer(modifier = Modifier.height(8.dp))
            }
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(onClick = {
                    if (context.hasCameraPermission()) {
                        captureRequested = true
                    } else {
                        pendingCapture = true
                        cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
                    }
                }) {
                    Text("Capture with Camera")
                }
                OutlinedButton(onClick = { galleryLauncher.launch("image/*") }) {
                    Text("Select from Gallery")
                }
            }
            Spacer(modifier = Modifier.height(8.dp))
            if (previewBytes != null) {
                Text("Preview", style = MaterialTheme.typography.titleMedium)
                Spacer(modifier = Modifier.height(6.dp))
                val previewBitmap = remember(previewBytes) {
                    android.graphics.BitmapFactory.decodeByteArray(previewBytes, 0, previewBytes!!.size)
                }
                Image(
                    bitmap = previewBitmap.asImageBitmap(),
                    contentDescription = "Captured preview",
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(220.dp)
                )
                Spacer(modifier = Modifier.height(8.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(
                        onClick = {
                            val bytes = previewBytes ?: return@Button
                            val envCode = nextShot?.envCode ?: return@Button
                            viewModel.submitShot(envCode, true, bytes)
                            previewBytes = null
                        },
                        enabled = !isSubmitting
                    ) {
                        Text("Accept")
                    }
                    OutlinedButton(
                        onClick = {
                            val bytes = previewBytes ?: return@OutlinedButton
                            val envCode = nextShot?.envCode ?: return@OutlinedButton
                            viewModel.submitShot(envCode, false, bytes)
                            previewBytes = null
                        },
                        enabled = !isSubmitting
                    ) {
                        Text("Reject")
                    }
                }
            }
            Spacer(modifier = Modifier.height(16.dp))
            if (forceComplete) {
                Text("Finish the remaining shots to continue.")
            } else {
                OutlinedButton(onClick = onBack) {
                    Text("Back to Objects")
                }
            }
        }
    }

    if (isSubmitting) {
        BlockingLoadingDialog("Saving shot...")
    }
}

@Composable
private fun TrainingSelectionScreen(
    envSpec: List<EnvSpecItem>,
    trainingViewModel: TrainingViewModel,
    onBack: () -> Unit
) {
    // The user must confirm 3 environments, one at a time.
    val selectionState by trainingViewModel.selectionState.collectAsState()
    val isSubmitting by trainingViewModel.isSubmitting.collectAsState()
    val nonClean = envSpec.filterNot { it.isClean || it.envCode.equals("Clean", ignoreCase = true) }
    val selected = selectionState.selectedEnvCodes
    val remaining = selectionState.remainingToSelect
    val selectionComplete = remaining == 0
    var pendingConfirm by remember { mutableStateOf<EnvSpecItem?>(null) }

    LaunchedEffect(Unit) {
        trainingViewModel.loadSelection()
    }

    Column(modifier = Modifier.fillMaxSize().padding(16.dp)) {
        Text("Training Environment Selection", style = MaterialTheme.typography.titleLarge)
        Spacer(modifier = Modifier.height(8.dp))
        Text("Select 3 non-Clean environments before creating any objects.")
        Spacer(modifier = Modifier.height(4.dp))
        Text("Selections are locked after capture starts.")
        Spacer(modifier = Modifier.height(6.dp))
        Text("Remaining selections: $remaining")
        Spacer(modifier = Modifier.height(12.dp))
        LazyColumn(modifier = Modifier.weight(1f, fill = true)) {
            items(nonClean) { env ->
                val isSelected = selected.contains(env.envCode)
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clickable(enabled = !selectionComplete && !isSelected && !isSubmitting) {
                            pendingConfirm = env
                        }
                        .padding(vertical = 6.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Checkbox(
                        checked = isSelected,
                        onCheckedChange = {
                            if (!isSelected && !selectionComplete && !isSubmitting) {
                                pendingConfirm = env
                            }
                        },
                        enabled = !selectionComplete
                    )
                    Spacer(modifier = Modifier.size(8.dp))
                    Column {
                        Text(env.envCode, fontWeight = FontWeight.SemiBold)
                        Text(env.friendlyName.ifBlank { env.envName }.ifBlank { "Unnamed" })
                    }
                }
                Divider()
            }
        }
        Spacer(modifier = Modifier.height(8.dp))
        OutlinedButton(onClick = onBack) {
            Text("Back")
        }
    }

    if (pendingConfirm != null) {
        val env = pendingConfirm!!
        AlertDialog(
            onDismissRequest = { pendingConfirm = null },
            title = { Text("Confirm environment") },
            text = {
                Text(
                    "Add ${env.envCode} (${env.friendlyName.ifBlank { env.envName }.ifBlank { "Unnamed" }})?"
                )
            },
            confirmButton = {
                Button(
                    onClick = {
                        trainingViewModel.confirmSelection(env.envCode)
                        pendingConfirm = null
                    },
                    enabled = !isSubmitting
                ) {
                    Text("Confirm")
                }
            },
            dismissButton = {
                OutlinedButton(onClick = { pendingConfirm = null }) {
                    Text("Cancel")
                }
            }
        )
    }

    if (isSubmitting) {
        BlockingLoadingDialog("Saving training selection...")
    }
}

@Composable
private fun TrainingListsScreen(
    trainingViewModel: TrainingViewModel,
    onBack: () -> Unit
) {
    val trainingLists by trainingViewModel.trainingLists.collectAsState()
    val isSubmitting by trainingViewModel.isSubmitting.collectAsState()

    LaunchedEffect(Unit) {
        trainingViewModel.loadTrainingLists()
    }

    Column(modifier = Modifier.fillMaxSize().padding(16.dp)) {
        Text("Training Lists", style = MaterialTheme.typography.titleLarge)
        Spacer(modifier = Modifier.height(8.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = trainingViewModel::loadTrainingLists, enabled = !isSubmitting) {
                Text("Refresh")
            }
            OutlinedButton(onClick = onBack) { Text("Back") }
        }
        Spacer(modifier = Modifier.height(12.dp))
        LazyColumn(modifier = Modifier.weight(1f, fill = true)) {
            items(trainingLists) { list ->
                Column(modifier = Modifier.padding(vertical = 6.dp)) {
                    Text("Training file: ${list.trainingFile ?: "Unknown"}")
                    Text("Object: ${list.objectId ?: "Unknown"}")
                    list.shots.forEach { shot ->
                        Text("- ${shot.envCode} (${shot.sequence ?: 0})")
                    }
                    if (list.postTrainingShots.isNotEmpty()) {
                        Spacer(modifier = Modifier.height(4.dp))
                        Text("PostTraining shots:")
                        list.postTrainingShots.forEach { shot ->
                            Text("- ${shot.envCode} (${shot.sequence ?: 0})")
                        }
                    }
                }
                Divider()
            }
        }
    }

    if (isSubmitting) {
        BlockingLoadingDialog("Loading training lists...")
    }
}

@Composable
private fun ThresholdSettingsScreen(
    sessionViewModel: SessionViewModel,
    backendUrl: String,
    onBack: () -> Unit
) {
    val thresholdConfig by sessionViewModel.thresholds.collectAsState()
    val isLoading by sessionViewModel.isLoading.collectAsState()
    var thresholdError by remember { mutableStateOf<String?>(null) }
    var confInput by remember(thresholdConfig) {
        mutableStateOf(thresholdConfig?.confThreshold?.toString() ?: "0.8")
    }
    var marginInput by remember(thresholdConfig) {
        mutableStateOf(thresholdConfig?.marginThreshold?.toString() ?: "0.1")
    }

    LaunchedEffect(backendUrl) {
        sessionViewModel.loadThresholds()
    }

    Column(modifier = Modifier.fillMaxSize().padding(16.dp)) {
        Text("Decision Thresholds", style = MaterialTheme.typography.titleLarge)
        Spacer(modifier = Modifier.height(8.dp))
        Text("These control when the model says \"I don't know\".")
        Text("Once confirmed, the values lock to keep results consistent.")
        Spacer(modifier = Modifier.height(12.dp))

        Text("How they work", style = MaterialTheme.typography.titleMedium)
        Text("T_conf: Minimum confidence required to accept a match.")
        Text("T_margin: Minimum gap between #1 and #2 (0 disables).")
        Text("If either rule fails, the model says \"I don't know\".")
        Spacer(modifier = Modifier.height(6.dp))
        val thresholdsLocked = thresholdConfig?.locked == true
        OutlinedTextField(
            value = confInput,
            onValueChange = { confInput = it },
            label = { Text("T_conf / epsilon (0.0 - 1.0)") },
            enabled = !thresholdsLocked,
            modifier = Modifier.fillMaxWidth()
        )
        Spacer(modifier = Modifier.height(6.dp))
        OutlinedTextField(
            value = marginInput,
            onValueChange = { marginInput = it },
            label = { Text("T_margin (0.0 - 1.0, 0 disables)") },
            enabled = !thresholdsLocked,
            modifier = Modifier.fillMaxWidth()
        )
        Spacer(modifier = Modifier.height(6.dp))
        if (thresholdsLocked) {
            Text("Decision thresholds are locked.")
        } else {
            Button(
                onClick = {
                    val conf = confInput.toDoubleOrNull()
                    val margin = marginInput.toDoubleOrNull()
                    if (conf == null || margin == null || conf !in 0.0..1.0 || margin !in 0.0..1.0) {
                        thresholdError = "Use numbers between 0.0 and 1.0."
                    } else {
                        thresholdError = null
                        sessionViewModel.setThresholds(conf, margin)
                    }
                }
            ) {
                Text("Set & Lock Thresholds")
            }
        }
        if (thresholdError != null) {
            Spacer(modifier = Modifier.height(6.dp))
            Text(text = thresholdError ?: "", color = MaterialTheme.colorScheme.error)
        }

        Spacer(modifier = Modifier.height(12.dp))
        OutlinedButton(onClick = onBack) { Text("Back") }
    }

    if (isLoading) {
        BlockingLoadingDialog("Saving settings...")
    }
}

@Composable
private fun InferenceScreen(
    inferenceViewModel: InferenceViewModel,
    onBack: () -> Unit,
    onStartSequence: (String, String) -> Unit
) {
    val uiState by inferenceViewModel.uiState.collectAsState()
    val objects by inferenceViewModel.objects.collectAsState()
    val errorMessage by inferenceViewModel.errorMessage.collectAsState()
    val isSubmitting by inferenceViewModel.isSubmitting.collectAsState()
    val thresholds by inferenceViewModel.thresholds.collectAsState()
    val lastPrediction by inferenceViewModel.lastPrediction.collectAsState()
    val inferenceMode by inferenceViewModel.inferenceMode.collectAsState()
    val context = LocalContext.current
    var previewBytes by remember { mutableStateOf<ByteArray?>(null) }
    var captureRequested by remember { mutableStateOf(false) }
    var pendingCapture by remember { mutableStateOf(false) }
    var pendingConfirm by remember { mutableStateOf<ObjectRecord?>(null) }
    var newObjectName by remember { mutableStateOf("") }
    val galleryLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri ->
        previewBytes = uri?.let { context.readBytesFromUri(it) }
    }
    val cameraPermissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted && pendingCapture) {
            captureRequested = true
        }
        pendingCapture = false
    }

    Column(modifier = Modifier.fillMaxSize().padding(16.dp)) {
        Text("Inference (Phase 2)", style = MaterialTheme.typography.titleLarge)
        Spacer(modifier = Modifier.height(8.dp))
        LaunchedEffect(Unit) {
            inferenceViewModel.loadInferenceMode()
        }
        val modeValue = inferenceMode?.mode?.trim()?.lowercase()
        val modeLabel = when (modeValue) {
            "semi-automatic" -> "Semi-Automatic"
            "full-automatic" -> "Full-Automatic"
            else -> null
        }
        if (modeLabel == null) {
            Text("Choose inference mode before the first inference:")
            Spacer(modifier = Modifier.height(6.dp))
            Text("Semi-Automatic: you confirm each accepted prediction.")
            Text("Full-Automatic: accepted predictions are auto-saved to the support set.")
            Spacer(modifier = Modifier.height(8.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(
                    onClick = { inferenceViewModel.setInferenceMode("semi-automatic") },
                    enabled = !isSubmitting
                ) {
                    Text("Semi-Automatic")
                }
                OutlinedButton(
                    onClick = { inferenceViewModel.setInferenceMode("full-automatic") },
                    enabled = !isSubmitting
                ) {
                    Text("Full-Automatic")
                }
            }
            Spacer(modifier = Modifier.height(12.dp))
        } else {
            Text("Inference mode: $modeLabel")
            Spacer(modifier = Modifier.height(8.dp))
        }
        val showSummary = lastPrediction != null &&
            uiState !is InferenceUiState.Idle &&
            uiState !is InferenceUiState.RunningInference
        if (showSummary) {
            InferenceConfigSummary(
                prediction = lastPrediction!!,
                thresholds = thresholds
            )
            Spacer(modifier = Modifier.height(12.dp))
        }
        when (val state = uiState) {
            InferenceUiState.Idle,
            InferenceUiState.RunningInference -> {
                CameraCapture(
                    captureTrigger = captureRequested,
                    onImageCaptured = { bytes ->
                        previewBytes = bytes
                        captureRequested = false
                    },
                    onError = { captureRequested = false }
                )
                Spacer(modifier = Modifier.height(8.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(onClick = {
                        if (context.hasCameraPermission()) {
                            captureRequested = true
                        } else {
                            pendingCapture = true
                            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
                        }
                    }) {
                        Text("Capture with Camera")
                    }
                    OutlinedButton(onClick = { galleryLauncher.launch("image/*") }) {
                        Text("Select from Gallery")
                    }
                }
                if (previewBytes != null) {
                    Spacer(modifier = Modifier.height(8.dp))
                    val previewBitmap = remember(previewBytes) {
                        android.graphics.BitmapFactory.decodeByteArray(previewBytes, 0, previewBytes!!.size)
                    }
                    Image(
                        bitmap = previewBitmap.asImageBitmap(),
                        contentDescription = "Inference preview",
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(220.dp)
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        Button(
                            onClick = {
                                val bytes = previewBytes ?: return@Button
                                inferenceViewModel.runInference(bytes)
                                previewBytes = null
                            },
                            enabled = !isSubmitting && modeLabel != null
                        ) {
                            Text("Run Inference")
                        }
                        OutlinedButton(
                            onClick = { previewBytes = null },
                            enabled = !isSubmitting
                        ) {
                            Text("Discard")
                        }
                    }
                }
            }

            is InferenceUiState.ConfirmPrediction -> {
                Text("Model prediction:")
                Spacer(modifier = Modifier.height(6.dp))
                Text(state.prediction.objectName ?: "Unknown")
                Spacer(modifier = Modifier.height(6.dp))
                state.prediction.probsTopK.forEach { item ->
                    Text("- ${item.objectName ?: "Unknown"}: ${"%.2f".format(item.prob)}")
                }
                Spacer(modifier = Modifier.height(12.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(
                        onClick = { inferenceViewModel.confirmPrediction(true) },
                        enabled = !isSubmitting
                    ) {
                        Text("Confirm")
                    }
                    OutlinedButton(
                        onClick = { inferenceViewModel.confirmPrediction(false) },
                        enabled = !isSubmitting
                    ) {
                        Text("Reject")
                    }
                }
                Spacer(modifier = Modifier.height(8.dp))
                OutlinedButton(
                    onClick = inferenceViewModel::cancelPending,
                    enabled = !isSubmitting
                ) {
                    Text("Cancel")
                }
            }

            is InferenceUiState.AutoAccepted -> {
                Text("Auto-accepted prediction.")
                Spacer(modifier = Modifier.height(6.dp))
                Text("Saved to: ${state.prediction.objectName ?: "Unknown"}")
                Spacer(modifier = Modifier.height(8.dp))
                Button(onClick = inferenceViewModel::resetToIdle, enabled = !isSubmitting) {
                    Text("Continue")
                }
            }

            is InferenceUiState.RejectPredictionChoice -> {
                Text("Prediction rejected.")
                Spacer(modifier = Modifier.height(6.dp))
                Text("Is this a new object or an existing one?")
                Spacer(modifier = Modifier.height(12.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(
                        onClick = { inferenceViewModel.chooseRejectedPrediction(false) },
                        enabled = !isSubmitting
                    ) {
                        Text("Existing Object")
                    }
                    OutlinedButton(
                        onClick = { inferenceViewModel.chooseRejectedPrediction(true) },
                        enabled = !isSubmitting
                    ) {
                        Text("New Object")
                    }
                }
                Spacer(modifier = Modifier.height(8.dp))
                OutlinedButton(
                    onClick = inferenceViewModel::cancelPending,
                    enabled = !isSubmitting
                ) {
                    Text("Cancel")
                }
            }

            is InferenceUiState.AskNewOrExisting -> {
                Text("Low confidence result.")
                Spacer(modifier = Modifier.height(6.dp))
                Text("Is this a new object?")
                Spacer(modifier = Modifier.height(12.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(
                        onClick = { inferenceViewModel.decideUnknown(true) },
                        enabled = !isSubmitting
                    ) {
                        Text("New Object")
                    }
                    OutlinedButton(
                        onClick = { inferenceViewModel.decideUnknown(false) },
                        enabled = !isSubmitting
                    ) {
                        Text("Existing Object")
                    }
                }
                Spacer(modifier = Modifier.height(8.dp))
                OutlinedButton(
                    onClick = inferenceViewModel::cancelPending,
                    enabled = !isSubmitting
                ) {
                    Text("Cancel")
                }
            }

            is InferenceUiState.Correction -> {
                Text("Select the correct object:")
                Spacer(modifier = Modifier.height(8.dp))
                LazyColumn(modifier = Modifier.weight(1f, fill = true)) {
                    items(objects) { record ->
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clickable { pendingConfirm = record }
                                .padding(vertical = 6.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(record.objectName.ifBlank { record.objectId }, fontWeight = FontWeight.SemiBold)
                        }
                        Divider()
                    }
                }
                Spacer(modifier = Modifier.height(8.dp))
                OutlinedButton(
                    onClick = inferenceViewModel::cancelPending,
                    enabled = !isSubmitting
                ) {
                    Text("Cancel")
                }
            }

            is InferenceUiState.NewObjectName -> {
                Text("Enter a name for the new object:")
                Spacer(modifier = Modifier.height(8.dp))
                OutlinedTextField(
                    value = newObjectName,
                    onValueChange = { newObjectName = it },
                    label = { Text("Object name") },
                    modifier = Modifier.fillMaxWidth()
                )
                Spacer(modifier = Modifier.height(8.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(
                        onClick = {
                            inferenceViewModel.createObjectFromPending(newObjectName)
                            newObjectName = ""
                        },
                        enabled = newObjectName.isNotBlank() && !isSubmitting
                    ) {
                        Text("Create")
                    }
                    OutlinedButton(
                        onClick = inferenceViewModel::cancelPending,
                        enabled = !isSubmitting
                    ) {
                        Text("Cancel")
                    }
                }
            }

            is InferenceUiState.AskStartSequence -> {
                Text("Capture 4 more shots for ${state.objectName}.")
                Spacer(modifier = Modifier.height(6.dp))
                Text("You need: 1 Clean + the 3 selected environments.")
                Spacer(modifier = Modifier.height(12.dp))
                Button(
                    onClick = {
                        inferenceViewModel.resetToIdle()
                        onStartSequence(state.objectId, state.objectName)
                    }
                ) {
                    Text("Start Capture")
                }
            }
        }

        if (errorMessage != null) {
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                text = errorMessage ?: "",
                color = MaterialTheme.colorScheme.error,
                style = MaterialTheme.typography.bodyMedium
            )
        }
        Spacer(modifier = Modifier.height(12.dp))
        if (uiState !is InferenceUiState.AskStartSequence) {
            OutlinedButton(onClick = {
                inferenceViewModel.cancelPending()
                onBack()
            }) {
                Text("Back")
            }
        }
    }

    if (pendingConfirm != null) {
        val record = pendingConfirm!!
        AlertDialog(
            onDismissRequest = { pendingConfirm = null },
            title = { Text("Confirm object") },
            text = { Text("Save as ${record.objectName.ifBlank { record.objectId }}?") },
            confirmButton = {
                Button(
                    onClick = {
                        inferenceViewModel.submitCorrection(record.objectId, true)
                        pendingConfirm = null
                    },
                    enabled = !isSubmitting
                ) {
                    Text("Confirm")
                }
            },
            dismissButton = {
                OutlinedButton(onClick = { pendingConfirm = null }) {
                    Text("Cancel")
                }
            }
        )
    }

    if (isSubmitting) {
        val loadingMessage = when (uiState) {
            InferenceUiState.RunningInference -> "Loading FEAT model and running inference..."
            is InferenceUiState.ConfirmPrediction -> "Saving confirmation..."
            is InferenceUiState.RejectPredictionChoice -> "Processing decision..."
            is InferenceUiState.AskNewOrExisting -> "Processing decision..."
            is InferenceUiState.Correction -> "Saving correction..."
            is InferenceUiState.NewObjectName -> "Creating object..."
            is InferenceUiState.AskStartSequence -> "Preparing capture sequence..."
            else -> "Working..."
        }
        BlockingLoadingDialog(loadingMessage)
    }
}

@Composable
private fun InferenceConfigSummary(
    prediction: InferencePrediction,
    thresholds: ThresholdConfigResponse?
) {
    val featConfidence = prediction.featConfidence ?: prediction.maxProb
    Text("FEAT confidence: ${"%.2f".format(featConfidence)}")
    if (thresholds == null) {
        Text("Threshold settings not available.")
        return
    }
    Spacer(modifier = Modifier.height(6.dp))
    Text("Thresholds used:")
    Text("T_conf: ${"%.2f".format(thresholds.confThreshold)} (min confidence to accept).")
    Text("T_margin: ${"%.2f".format(thresholds.marginThreshold)} (min gap vs #2; 0 disables).")
}

private fun Context.readBytesFromUri(uri: Uri): ByteArray? {
    val stream: InputStream = contentResolver.openInputStream(uri) ?: return null
    return stream.use { it.readBytes() }
}

private fun Context.hasCameraPermission(): Boolean =
    ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
        android.content.pm.PackageManager.PERMISSION_GRANTED
